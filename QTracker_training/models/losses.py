import itertools

import tensorflow as tf
from typing import Callable

OVERLAP_LAMBDA = 0.1
DISTANCE_LAMBDA = 5e-4
EPSILON = 1e-7


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


def min_perm_loss(n_pairs: int) -> Callable:
    """
    Permutation-invariant loss for a fixed-N multi-track finder.

    Brute-forces all N! permutations of predicted track slots vs. ground-truth
    track slots and picks the minimum-cost assignment, so track slot ordering
    does not affect training.

    For N=1, the computed scalar is mathematically identical to ``custom_loss``
    when called with equivalently shaped inputs (``y_true: (B,1,2,62)``,
    ``y_pred: (B,1,2,62,201)`` vs the single-track ``(B,2,62)``/``(B,2,62,201)``).
    Note that ``custom_loss`` and ``min_perm_loss(1)`` are not interchangeable
    at the call site — inputs must carry the N=1 pair axis.

    Args:
        n_pairs (int): Number of dimuon pairs N. Must satisfy 1 <= n_pairs <= 3
            (max 6 permutations at N=3).

    Returns:
        A loss function with signature ``loss(y_true, y_pred) -> tf.Tensor``.

        Where:
            y_true: Ground truth tensor with shape ``(B, N, 2, 62)``. Integer
                element IDs (0 = no hit); axis 2 holds mu+/mu-.
            y_pred: Predicted softmax probabilities with shape
                ``(B, N, 2, 62, 201)``; axis 2 holds mu+/mu-.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if n_pairs > 3:
        raise ValueError(
            f"n_pairs > 3 not supported (would generate {__import__('math').factorial(n_pairs)} "
            f"permutations); got {n_pairs}"
        )

    # Precompute permutation tensors once at factory time (not per batch).
    perms = list(itertools.permutations(range(n_pairs)))
    perm_tensors = [tf.constant(list(p), dtype=tf.int32) for p in perms]

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true (tf.Tensor): Shape ``(B, N, 2, 62)``.
            y_pred (tf.Tensor): Shape ``(B, N, 2, 62, 201)``.

        Returns:
            tf.Tensor: Scalar loss value.
        """
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.int32)

        # --- Cost matrix: cost[b, i, j] = CE cost of assigning pred slot i to GT slot j ---
        # Expand pred: (B, N, 1, 2, 62, 201)
        pred_expand = tf.expand_dims(y_pred, axis=2)
        # Expand true: (B, 1, N, 2, 62)
        true_expand = tf.expand_dims(y_true, axis=1)

        # CE per (pred slot, GT slot, muon, detector): (B, N, N, 2, 62)
        #   axis 3 = muon dim (size 2), axis 4 = detector dim (size 62)
        ce = tf.keras.losses.sparse_categorical_crossentropy(true_expand, pred_expand)

        # Sum over muon dim (axis=3): (B, N, N, 62)
        # Mean over detector dim (now axis=3): (B, N, N)
        cost = tf.reduce_mean(tf.reduce_sum(ce, axis=3), axis=3)

        # --- Find minimum-cost permutation ---
        perm_costs = []
        for perm_tensor in perm_tensors:
            # Reorder columns of cost by permutation, then trace = sum_i cost[b, i, perm[i]]
            cost_perm = tf.gather(cost, perm_tensor, axis=2)  # (B, N, N) cols reordered
            perm_costs.append(tf.linalg.trace(cost_perm))  # (B,)

        # Stack along axis=1 then take minimum: (B,)
        min_cost = tf.reduce_min(tf.stack(perm_costs, axis=1), axis=1)

        # --- Overlap penalty: discourage mu+/mu- collapsing to same position ---
        p_plus = y_pred[:, :, 0, :, :]  # (B, N, 62, 201)
        p_minus = y_pred[:, :, 1, :, :]  # (B, N, 62, 201)
        # Sum over 201 element IDs per (pair, detector): (B, N, 62)
        overlap = tf.reduce_sum(tf.square(p_plus - p_minus), axis=-1)

        # Mean over detectors (axis=2), sum over N pairs (axis=1): (B,)
        # For N=1 this reduces to mean_d(sum_c (p+−p−)²), matching custom_loss exactly.
        overlap_penalty = OVERLAP_LAMBDA * tf.reduce_sum(
            tf.reduce_mean(overlap, axis=2), axis=1
        )

        return tf.reduce_mean(min_cost + overlap_penalty)

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
