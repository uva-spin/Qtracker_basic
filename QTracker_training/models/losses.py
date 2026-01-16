import tensorflow as tf
from typing import Callable

OVERLAP_LAMBDA = 0.1
DISTANCE_LAMBDA = 5e-4


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
        bce = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        weights = 1 + (pos_weight - 1) * y_true
        bce_loss = bce(y_true, y_pred, sample_weight=weights)
        return tf.reduce_mean(bce_loss)

    return loss
