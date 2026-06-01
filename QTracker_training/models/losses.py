from typing import Callable

import numpy as np
import tensorflow as tf

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
    mup_station2_weight: float = 1.0,
) -> Callable:
    """
    Multi-track segmentation loss.

    Components:
      1) Masked sparse categorical cross-entropy on nonzero targets.
      2) Weighted binary cross-entropy for hit presence.

    Args:
        lambda_presence: weight for presence term
        pos_weight_presence: weight multiplier for positive presence labels
        mup_station2_weight: extra loss weight for mu+ at Station 2 detectors
            (det 19-30, 0-indexed 18-29). Use >1.0 to focus learning on the
            mu+ Station 2 asymmetry. 1.0 = no effect (default).

    Returns:
        Loss function.
    """
    # Per-position weight tensor shape (1, 1, 2, 62) — broadcasts over (B, P, 2, 62).
    # Axis 2 index 0 = mu+, index 1 = mu-.
    # Station 2 = detectors 19-30 (1-indexed) = indices 18-29 (0-indexed).
    _w = np.ones((1, 1, 2, 62), dtype=np.float32)
    _w[0, 0, 0, 18:30] = mup_station2_weight  # upweight mu+ Station 2 only
    _det_weights = tf.constant(_w, dtype=tf.float32)

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

        # Apply per-detector weights before summing; normalize by weighted hit count
        # so the scale stays comparable regardless of mup_station2_weight.
        ce_masked = ce * mask_hit * _det_weights

        # Normalize by number of true hits
        num_hits = tf.reduce_sum(mask_hit * _det_weights)  # (B,P,2,62) -> scalar
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


# ---------------------------------------------------------------------------
# Confidence head losses (Proposals A and B)
# ---------------------------------------------------------------------------


def compute_track_f1(
    seg_pred: tf.Tensor,
    gt_labels: tf.Tensor,
) -> tf.Tensor:
    """Compute per-event F1 overlap between predicted and ground-truth tracks.

    This implements the overlap metric described in Proposal B.  For every
    event the predicted element-IDs (argmax of *seg_pred*) are compared with
    the ground-truth element-IDs in *gt_labels*.  A hit is considered a
    true-positive when the predicted element-ID equals the ground-truth
    element-ID **and** both are non-zero (element-ID 0 encodes "no hit").

    The F1 score is computed jointly over both muon charges (μ⁺ and μ⁻)
    so that the single scalar reflects the overall quality of the dimuon
    reconstruction.

    Args:
        seg_pred: Segmentation softmax output of shape ``(B, 2, 62, C)``
            where *C* is the number of element-ID classes (typically 201).
        gt_labels: Ground-truth element-IDs of shape ``(B, 2, 62)`` with
            integer values in ``[0, C)``.

    Returns:
        A float32 tensor of shape ``(B,)`` containing the F1 score for each
        event.  Values are in ``[0, 1]``.
    """

    # Predicted element-IDs via argmax (not differentiable – that is fine
    # because this tensor is only used as a *target*, never back-propagated).
    pred_hits = tf.cast(tf.argmax(seg_pred, axis=-1), tf.int32)  # (B, 2, 62)
    gt_hits = tf.cast(gt_labels, tf.int32)  # (B, 2, 62)

    pred_nonzero = tf.cast(tf.not_equal(pred_hits, 0), tf.float32)
    true_nonzero = tf.cast(tf.not_equal(gt_hits, 0), tf.float32)

    # A match requires *both* sides to be non-zero and to agree on the
    # element-ID.
    match = (
        tf.cast(tf.equal(pred_hits, gt_hits), tf.float32) * pred_nonzero * true_nonzero
    )

    # Aggregate over muon charge and detector axes → per-event scalars.
    precision = tf.reduce_sum(match, axis=[1, 2]) / (
        tf.reduce_sum(pred_nonzero, axis=[1, 2]) + EPSILON
    )
    recall = tf.reduce_sum(match, axis=[1, 2]) / (
        tf.reduce_sum(true_nonzero, axis=[1, 2]) + EPSILON
    )
    f1 = 2.0 * precision * recall / (precision + recall + EPSILON)

    return f1  # (B,)


def confidence_bce(
    confidence_weight: float = 0.1,
    pos_weight: float = 1.0,
) -> Callable:
    """Binary cross-entropy loss for the **event-level** confidence head
    (Proposal A – "Stop-or-Go").

    The target is a scalar label per event:
      * **1** – at least one valid track is present in the event.
      * **0** – no valid tracks remain.

    An optional *pos_weight* > 1 up-weights the positive class so that the
    model is biased towards recall (it is safer to predict "tracks present"
    than to miss them).

    The returned scalar is **pre-multiplied** by *confidence_weight* so that
    the caller can add it directly to the total loss without an extra
    ``loss_weights`` entry.

    Args:
        confidence_weight: Multiplicative factor applied to the loss before
            returning.  Use this to balance the confidence objective against
            the denoising and segmentation losses.
        pos_weight: Up-weighting factor for the positive class (label = 1).

    Returns:
        A loss function with signature ``(y_true, y_pred) -> scalar``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Binary presence labels of shape ``(B, 1)`` or ``(B,)``.
            y_pred: Sigmoid confidence scores of shape ``(B, 1)`` or ``(B,)``.
        """
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32)

        # Per-sample weights: up-weight positives to improve recall.
        sample_weights = 1.0 + (pos_weight - 1.0) * y_true

        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (B,)
        weighted = bce * tf.squeeze(sample_weights, axis=-1)

        return confidence_weight * tf.reduce_mean(weighted)

    return loss


def confidence_f1_loss(
    seg_pred: tf.Tensor,
    confidence_pred: tf.Tensor,
    gt_labels: tf.Tensor,
    confidence_weight: float = 0.1,
) -> tf.Tensor:
    """Confidence loss for Proposal B – "Track Correctness".

    Computes a soft F1 target from the segmentation predictions and the
    ground-truth labels, then trains the confidence head to regress that
    target via binary cross-entropy.

    This function is intended to be called **inside a custom ``train_step``**
    because it requires simultaneous access to the segmentation output, the
    confidence output, and the ground-truth segmentation labels – something
    that is not possible with Keras' standard per-output loss API.

    Args:
        seg_pred: Segmentation softmax output, shape ``(B, 2, 62, C)``.
        confidence_pred: Confidence sigmoid output, shape ``(B, 1)``.
        gt_labels: Ground-truth element-IDs, shape ``(B, 2, 62)``.
        confidence_weight: Multiplicative scaling factor for the loss.

    Returns:
        Scalar loss tensor.
    """

    f1_target = tf.stop_gradient(compute_track_f1(seg_pred, gt_labels))  # (B,)
    f1_target = tf.reshape(f1_target, (-1, 1))  # (B, 1)

    confidence_pred = tf.cast(tf.reshape(confidence_pred, (-1, 1)), tf.float32)

    bce = tf.keras.losses.binary_crossentropy(f1_target, confidence_pred)  # (B,)

    return confidence_weight * tf.reduce_mean(bce)
