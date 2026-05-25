import itertools

import tensorflow as tf
from typing import Callable

OVERLAP_LAMBDA = 0.1
DISTANCE_LAMBDA = 5e-4
EPSILON = 1e-7

MAX_PERMS_P = (
    5  # P (max_pairs) should not exceed this; beyond that, P! becomes impractical.
)


def _build_perm_table(p: int) -> tf.Tensor:
    """Build permutation index table for a given number of slots."""
    if p > MAX_PERMS_P:
        raise ValueError(
            f"max_pairs={p} exceeds limit of {MAX_PERMS_P} "
            f"(would require {p}! = {len(list(itertools.permutations(range(p))))} permutations)"
        )
    return tf.constant(list(itertools.permutations(range(p))), dtype=tf.int32)


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


def min_perm_multi_track_loss(
    max_pairs: int = 5,
    lambda_presence: float = 1.0,
    pos_weight_presence: float = 5.0,
    focal_gamma: float = 2.0,
    lambda_diversity: float = 0.05,
) -> Callable:
    """
    Multi-track segmentation loss with min-over-permutations matching,
    focal presence, and diversity penalty.

    Uses brute-force enumeration of all P! permutations to find the optimal
    assignment between predicted slots and GT slots. For P<=5 (at most 120
    perms), this is negligible compute vs the model forward pass and avoids
    the tf.py_function required by Hungarian matching (which is brittle under
    MirroredStrategy multi-GPU training).

    Components:
      1) Min-over-permutations sparse categorical CE on nonzero targets.
      2) Focal BCE for hit presence (inside the permutation min).
      3) Inter-pair diversity penalty (outside the min — depends only on y_pred).

    Args:
        max_pairs: number of output pair slots (must not exceed MAX_PERMS_P=5)
        lambda_presence: weight for presence/focal term
        pos_weight_presence: weight multiplier for positive presence labels
        focal_gamma: gamma parameter for focal loss (0 = standard BCE)
        lambda_diversity: weight for inter-pair diversity penalty

    Returns:
        Loss function with signature (y_true, y_pred) -> scalar.
    """
    all_perms = _build_perm_table(max_pairs)  # (P!, P)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        y_true: (B, P, 2, 62)     — sparse element IDs (0 = no hit)
        y_pred: (B, P, 2, 62, C)  — softmax probabilities
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # --- Step 1: Evaluate loss for all P! permutations ---
        # Gather produces (B, P!, P, 2, 62); transpose to (P!, B, P, 2, 62)
        y_true_all = tf.transpose(
            tf.gather(y_true, all_perms, axis=1), perm=[1, 0, 2, 3, 4]
        )

        # Broadcast predictions: (1, B, P, 2, 62, C)
        y_pred_exp = tf.expand_dims(y_pred, 0)

        # Hit mask for all perms
        mask_all = tf.cast(tf.not_equal(y_true_all, 0), tf.float32)

        # Masked sparse CE: (120, B, P, 2, 62)
        ce_all = tf.keras.losses.sparse_categorical_crossentropy(y_true_all, y_pred_exp)
        ce_masked = ce_all * mask_all
        num_hits = tf.reduce_sum(mask_all, axis=[2, 3, 4])  # (120, B)
        cls_per_perm = tf.reduce_sum(ce_masked, axis=[2, 3, 4]) / (num_hits + EPSILON)

        # Focal presence loss per perm
        y_hit = mask_all
        p0 = tf.clip_by_value(y_pred_exp[..., 0], EPSILON, 1.0 - EPSILON)
        p_hit = 1.0 - p0
        p_t = y_hit * p_hit + (1.0 - y_hit) * (1.0 - p_hit)
        focal_weight = tf.pow(1.0 - p_t, focal_gamma)
        class_weight = 1.0 + (pos_weight_presence - 1.0) * y_hit
        bce = tf.keras.backend.binary_crossentropy(y_hit, p_hit)
        presence_per_perm = tf.reduce_mean(
            bce * focal_weight * class_weight, axis=[2, 3, 4]
        )  # (120, B)

        # --- Step 2: Min over permutations ---
        total_per_perm = cls_per_perm + lambda_presence * presence_per_perm
        best_loss = tf.reduce_min(total_per_perm, axis=0)  # (B,)
        matched_loss = tf.reduce_mean(best_loss)

        # --- Step 3: Diversity penalty (y_pred only, outside min) ---
        C = tf.shape(y_pred)[-1]
        elem_indices = tf.cast(tf.range(C), tf.float32)
        soft_ids = tf.reduce_sum(y_pred * elem_indices, axis=-1)  # (B, P, 2, 62)

        soft_i = tf.expand_dims(soft_ids, 2)
        soft_j = tf.expand_dims(soft_ids, 1)
        pairwise_dist = tf.reduce_sum(
            tf.square(soft_i - soft_j), axis=[-2, -1]
        )  # (B, P, P)

        P = tf.shape(soft_ids)[1]
        diag_mask = 1.0 - tf.eye(P, dtype=tf.float32)

        diversity_penalty = tf.reduce_sum(
            tf.exp(-pairwise_dist / 1000.0) * diag_mask
        ) / (
            tf.cast(P * (P - 1), tf.float32) * tf.cast(tf.shape(y_pred)[0], tf.float32)
            + EPSILON
        )

        return matched_loss + lambda_diversity * diversity_penalty

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
