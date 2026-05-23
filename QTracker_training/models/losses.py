import numpy as np
import tensorflow as tf
from typing import Callable
from scipy.optimize import linear_sum_assignment

OVERLAP_LAMBDA = 0.1
DISTANCE_LAMBDA = 5e-4
EPSILON = 1e-7
DIVERSITY_LAMBDA = 0.05


def custom_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function that combines sparse categorical cross-entropy for mu+ and mu- predictions
    with an overlap penalty to discourage overlapping predictions.

    Args:
        y_true (tf.Tensor): Ground truth tensor with shape (batch_size, 2, num_classes).
        y_pred (tf.Tensor): Predicted tensor with shape (batch_size, 2, num_classes).

    Returns:
        tf.Tensor: Computed loss value.
    """
    y_pred = tf.cast(y_pred, tf.float32)

    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(
        y_muPlus_true, y_muPlus_pred
    )
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(
        y_muMinus_true, y_muMinus_pred
    )

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    return tf.reduce_mean(loss_mup + loss_mum + OVERLAP_LAMBDA * overlap_penalty)


def _hungarian_match_np(cost_matrix: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = cost_matrix.shape[0]
    perm = np.arange(P, dtype=np.int32)
    perm[row_ind] = col_ind.astype(np.int32)
    return perm


def _compute_matching_indices(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute Hungarian matching for a batch of events.

    Args:
        y_true: (B, P, 2, 62) int32 ground truth element IDs
        y_pred: (B, P, 2, 62, C) float32 softmax predictions

    Returns:
        perms: (B, P) int32 tensor of GT permutation indices per event
    """

    def _match_single(args):
        yt, yp = args  # yt: (P, 2, 62), yp: (P, 2, 62, C)
        P = tf.shape(yt)[0]

        # Cost matrix: CE between each predicted slot i and GT slot j
        yt_exp = tf.cast(tf.expand_dims(yt, 0), tf.int32)  # (1, P, 2, 62)
        yp_exp = tf.expand_dims(yp, 1)  # (P, 1, 2, 62, C)

        # Tile for pairwise
        yt_tile = tf.tile(yt_exp, [P, 1, 1, 1])  # (P, P, 2, 62)
        yp_tile = tf.tile(yp_exp, [1, P, 1, 1, 1])  # (P, P, 2, 62, C)

        # Sparse CE per (pred_slot, gt_slot, charge, detector)
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            yt_tile, yp_tile
        )  # (P, P, 2, 62)

        # Sum over charge and detector dims to get (P, P) cost matrix
        cost = tf.reduce_sum(ce, axis=[-2, -1])  # (P, P)

        # Run Hungarian matching via numpy
        perm = tf.py_function(
            func=lambda c: _hungarian_match_np(c.numpy()),
            inp=[cost],
            Tout=tf.int32,
        )
        perm.set_shape([None])
        return perm

    # Map over batch
    perms = tf.map_fn(
        _match_single,
        (y_true, y_pred),
        fn_output_signature=tf.int32,
    )
    return perms


def hungarian_multi_track_loss(
    lambda_presence: float = 1.0,
    pos_weight_presence: float = 5.0,
    focal_gamma: float = 2.0,
    lambda_diversity: float = 0.05,
) -> Callable:
    """
    Multi-track segmentation loss with Hungarian matching, focal presence, and diversity penalty.

    Components:
      1) Hungarian-matched sparse categorical cross-entropy on nonzero targets.
      2) Focal BCE for hit presence (replaces vanilla weighted BCE).
      3) Inter-pair diversity penalty to prevent slot collapse.

    Args:
        lambda_presence: weight for presence/focal term (increased from 0.2 to 1.0)
        pos_weight_presence: weight multiplier for positive presence labels
        focal_gamma: gamma parameter for focal loss (0 = standard BCE)
        lambda_diversity: weight for inter-pair diversity penalty

    Returns:
        Loss function with signature (y_true, y_pred) -> scalar.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        y_true: (B, P, 2, 62)     — sparse element IDs (0 = no hit)
        y_pred: (B, P, 2, 62, C)  — softmax probabilities
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # --- Step 1: Hungarian matching ---
        perms = _compute_matching_indices(y_true, y_pred)  # (B, P)

        # Reorder y_true to match optimal assignment
        y_true_matched = tf.gather(y_true, perms, batch_dims=1)  # (B, P, 2, 62)

        # --- Step 2: Masked sparse CE (classification) ---
        mask_hit = tf.cast(tf.not_equal(y_true_matched, 0), tf.float32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_matched, y_pred
        )  # (B, P, 2, 62)
        ce_masked = ce * mask_hit
        num_hits = tf.reduce_sum(mask_hit)
        cls_loss = tf.reduce_sum(ce_masked) / (num_hits + EPSILON)

        # --- Step 3: Focal presence loss ---
        y_hit = mask_hit  # binary: does this position have a nonzero GT?
        p0 = tf.clip_by_value(y_pred[..., 0], EPSILON, 1.0 - EPSILON)
        p_hit = 1.0 - p0  # probability that a hit exists

        # Focal modulating factor: (1 - p_t)^gamma
        p_t = y_hit * p_hit + (1.0 - y_hit) * (1.0 - p_hit)
        focal_weight = tf.pow(1.0 - p_t, focal_gamma)

        # Class weighting (amplify positive examples)
        class_weight = 1.0 + (pos_weight_presence - 1.0) * y_hit

        bce = tf.keras.backend.binary_crossentropy(y_hit, p_hit)
        presence_loss = tf.reduce_mean(bce * focal_weight * class_weight)

        # --- Step 4: Inter-pair diversity penalty ---
        C = tf.shape(y_pred)[-1]
        elem_indices = tf.cast(tf.range(C), tf.float32)  # (C,)

        # Soft expected element ID per position
        soft_ids = tf.reduce_sum(y_pred * elem_indices, axis=-1)  # (B, P, 2, 62)

        # Pairwise L2 distance between all slot pairs
        soft_i = tf.expand_dims(soft_ids, 2)
        soft_j = tf.expand_dims(soft_ids, 1)
        pairwise_dist = tf.reduce_sum(
            tf.square(soft_i - soft_j), axis=[-2, -1]
        )  # (B, P, P)

        # Mask diagonal (self-pairs)
        P = tf.shape(soft_ids)[1]
        diag_mask = 1.0 - tf.eye(P, dtype=tf.float32)  # (P, P)

        # Penalty: encourage large distances between different slots
        diversity_penalty = tf.reduce_sum(
            tf.exp(-pairwise_dist / 1000.0) * diag_mask
        ) / (
            tf.cast(P * (P - 1), tf.float32) * tf.cast(tf.shape(y_pred)[0], tf.float32)
            + EPSILON
        )

        return (
            cls_loss
            + lambda_presence * presence_loss
            + lambda_diversity * diversity_penalty
        )

    return loss


def multi_track_loss(
    lambda_presence: float = 0.2,
    pos_weight_presence: float = 5.0,
) -> Callable:
    """
    Multi-track segmentation loss.

    Components:
      1) Masked sparse categorical cross-entropy on nonzero targets.
      2) Weighted binary cross-entropy for hit presence.

    Args:
        lambda_presence: weight for presence term
        pos_weight_presence: weight multiplier for positive presence labels

    Returns:
        Loss function.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        y_true: (B, P, 2, 62)
        y_pred: (B, P, 2, 62, C)
        """

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # --- Classification term (masked sparse CE) ---
        mask_hit = tf.cast(tf.not_equal(y_true, 0), tf.float32)  # mask for nonzero hits

        # Sparse CE per position
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )  # (B,P,2,62)

        ce_masked = ce * mask_hit

        # Normalize by number of true hits
        num_hits = tf.reduce_sum(mask_hit)  # (B,P,2,62) -> scalar
        cls_loss = tf.reduce_sum(ce_masked) / (num_hits + EPSILON)

        # --- Presence term (weighted BCE): penalize false positives and negatives ---
        y_hit = mask_hit  # binary hit presence (B,P,2,62)

        # probability of class 0 (no hit)
        p0 = tf.clip_by_value(y_pred[..., 0], EPSILON, 1.0 - EPSILON)

        # probability that a hit exists
        p_hit = 1.0 - p0  # shape (B,P,2,62)

        # class weighting (amplify positive examples)
        weights = (
            1.0 + (pos_weight_presence - 1.0) * y_hit
        )  # penalize false negatives more

        # compute BCE per position
        bce_loss = tf.keras.backend.binary_crossentropy(y_hit, p_hit)
        bce_loss_weighted = bce_loss * weights

        # normalize by sum of weights
        presence_loss = tf.reduce_sum(bce_loss_weighted) / (
            tf.reduce_sum(weights) + EPSILON
        )

        return cls_loss + lambda_presence * presence_loss

    return loss


def weighted_bce(pos_weight: float = 1.0) -> Callable:
    """
    Returns a weighted binary cross-entropy loss function. False negatives are penalized more heavily
    based on the provided positive weight (> 1).

    Args:
        pos_weight (float): Weight for positive class.

    Returns:
        A loss function that computes weighted binary cross-entropy.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        weights = 1 + (pos_weight - 1) * y_true
        bce_loss = bce(y_true, y_pred, sample_weight=weights)
        return tf.reduce_mean(bce_loss)

    return loss
