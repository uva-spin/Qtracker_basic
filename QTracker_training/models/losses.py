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
