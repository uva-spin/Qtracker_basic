"""Attention U-Net++ based denoiser + segmenter: end-to-end training

Supports three confidence modes via ``--confidence_mode``:

* **none** – original two-head model (denoise + segment).  Fully backward
  compatible with existing checkpoints and evaluation scripts.
* **event_level** – Proposal A "Stop-or-Go" confidence head.  Adds a
  third output that predicts whether *any* valid track remains in the
  current event state.  Trained with binary cross-entropy against a
  label derived from the ground-truth hit arrays (1 = tracks present,
  0 = no tracks).
* **track_quality** – Proposal B "Track Correctness" confidence head.
  Same architecture as *event_level* but the training target is the F1
  overlap between the model's own segmentation prediction and the
  best-matching ground-truth track.  Requires a custom ``train_step``
  because the target depends on the segmentation output.
"""

import argparse
import gc
import os

os.environ["TF_GPU_ALLOCATOR"] = (
    "cuda_malloc_async"  # Enable asynchronous GPU memory allocation for better performance
)

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
import tensorflow.keras.backend as K
from backbones import unetpp_backbone
from data_loader import load_data_denoise
from losses import (
    compute_track_f1,
    confidence_bce,
    confidence_f1_loss,
    custom_loss,
    weighted_bce,
)
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import AdamW

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Set mixed precision policy for better performance
mixed_precision.set_global_policy("mixed_float16")

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

# ---------------------------------------------------------------------------
# Valid confidence modes
# ---------------------------------------------------------------------------
CONFIDENCE_NONE = "none"
CONFIDENCE_EVENT_LEVEL = "event_level"  # Proposal A
CONFIDENCE_TRACK_QUALITY = "track_quality"  # Proposal B
_VALID_CONFIDENCE_MODES = {
    CONFIDENCE_NONE,
    CONFIDENCE_EVENT_LEVEL,
    CONFIDENCE_TRACK_QUALITY,
}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(
    num_detectors: int = 62,
    num_elementIDs: int = 201,
    use_bn: bool = False,
    dropout_bn: float = 0.0,
    dropout_enc: float = 0.0,
    denoise_base: int = 64,
    base: int = 64,
    use_attn: bool = False,
    use_attn_ffn: bool = True,
    dropout_attn: float = 0.0,
    confidence_mode: str = CONFIDENCE_NONE,
) -> tf.keras.Model:
    """Build the joint denoising + segmentation model, optionally with a
    confidence head.

    The confidence head branches off the segmentation backbone features
    (before the segmentation classification layer) and produces a single
    sigmoid scalar per event.  Its architecture is:

        GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3)
        → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)

    Args:
        num_detectors: Number of detectors (default: 62).
        num_elementIDs: Number of element IDs (default: 201).
        use_bn: Whether to use batch normalization.
        dropout_bn: Dropout rate for bottleneck layer.
        dropout_enc: Dropout rate for encoder blocks.
        denoise_base: Number of base channels in denoising U-Net++.
        base: Number of base channels in segmentation U-Net++.
        use_attn: Whether to use axial attention in the segmentation backbone.
        use_attn_ffn: Whether to use feed-forward layers in attention.
        dropout_attn: Dropout rate for attention block.
        confidence_mode: One of ``"none"``, ``"event_level"`` (Proposal A),
            or ``"track_quality"`` (Proposal B).  When not ``"none"`` a third
            output head is appended to the model.

    Returns:
        A ``tf.keras.Model`` instance.  Outputs are:
          * ``confidence_mode="none"``  → ``[denoise, segment]``
          * otherwise                   → ``[denoise, segment, confidence]``
    """

    if confidence_mode not in _VALID_CONFIDENCE_MODES:
        raise ValueError(
            f"Unknown confidence_mode={confidence_mode!r}.  "
            f"Choose from {_VALID_CONFIDENCE_MODES}."
        )

    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # ----- Denoising Backbone (first U-Net++) -----
    x = unetpp_backbone(
        input_layer,
        num_detectors,
        num_elementIDs,
        use_bn,
        dropout_bn,
        dropout_enc,
        denoise_base,
        use_attn=False,
    )

    denoise_out = layers.Conv2D(1, kernel_size=1, name="denoise", dtype=tf.float32)(x)

    # ----- Segmentation Backbone (second U-Net++) -----
    seg_features = unetpp_backbone(
        denoise_out,
        num_detectors,
        num_elementIDs,
        use_bn,
        dropout_bn,
        dropout_enc,
        base,
        use_attn,
        use_attn_ffn,
        dropout_attn,
    )

    # ----- Segmentation Head -----
    x = layers.Conv2D(2, kernel_size=1)(seg_features)
    x = layers.Permute((3, 1, 2))(x)  # (batch, 2, det, elem)
    seg_output = layers.Softmax(axis=-1, name="segment", dtype=tf.float32)(x)

    outputs = [denoise_out, seg_output]

    # ----- Confidence Head (optional) -----
    if confidence_mode != CONFIDENCE_NONE:
        # Tap into the segmentation backbone features (before the
        # classification conv) so that the confidence head has access to
        # rich spatial representations.
        conf = layers.GlobalAveragePooling2D()(seg_features)  # (B, base)
        conf = layers.Dense(128, activation="relu", dtype=tf.float32)(conf)
        conf = layers.Dropout(0.3)(conf)
        conf = layers.Dense(64, activation="relu", dtype=tf.float32)(conf)
        conf = layers.Dropout(0.3)(conf)
        # Single sigmoid output – shape (B, 1)
        conf = layers.Dense(
            1, activation="sigmoid", name="confidence", dtype=tf.float32
        )(conf)
        outputs.append(conf)

    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return model


# ---------------------------------------------------------------------------
# Custom Model subclass for Proposal B (track_quality)
# ---------------------------------------------------------------------------


class TrackFinderWithConfidence(tf.keras.Model):
    """Thin wrapper that overrides ``train_step`` / ``test_step`` so that the
    confidence head's target is the **F1 overlap** between the model's own
    segmentation prediction and the ground-truth labels.

    This is necessary because the F1 target depends on the segmentation
    output at each training step and therefore cannot be provided as a
    static label.

    Usage::

        functional_model = build_model(..., confidence_mode="track_quality")

        model = TrackFinderWithConfidence(
            inputs=functional_model.input,
            outputs=functional_model.outputs,
            denoise_loss_fn=weighted_bce(pos_weight=1.0),
            segment_loss_fn=custom_loss,
            confidence_weight=0.1,
            denoise_weight=10.0,
        )
        model.compile(optimizer=AdamW(...))
        model.fit(X, {"denoise": X_clean, "segment": y}, ...)
    """

    def __init__(
        self,
        *args,
        denoise_loss_fn=None,
        segment_loss_fn=None,
        confidence_weight: float = 0.1,
        denoise_weight: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if denoise_loss_fn is None:
            raise ValueError("denoise_loss_fn is required")
        if segment_loss_fn is None:
            raise ValueError("segment_loss_fn is required")
        self.denoise_loss_fn = denoise_loss_fn
        self.segment_loss_fn = segment_loss_fn
        self.confidence_weight = confidence_weight
        self.denoise_weight = denoise_weight

        # Tracked metrics
        self._loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._denoise_loss_tracker = tf.keras.metrics.Mean(name="denoise_loss")
        self._segment_loss_tracker = tf.keras.metrics.Mean(name="segment_loss")
        self._confidence_loss_tracker = tf.keras.metrics.Mean(name="confidence_loss")
        self._mean_f1_tracker = tf.keras.metrics.Mean(name="mean_f1")
        self._mean_conf_tracker = tf.keras.metrics.Mean(name="mean_confidence")

    @property
    def metrics(self):
        return [
            self._loss_tracker,
            self._denoise_loss_tracker,
            self._segment_loss_tracker,
            self._confidence_loss_tracker,
            self._mean_f1_tracker,
            self._mean_conf_tracker,
        ]

    # ---- helpers ---------------------------------------------------------

    def _compute_losses(self, y, outputs, training_label: str = ""):
        """Shared loss computation used by both train_step and test_step."""
        denoise_out, seg_out, confidence_out = outputs

        d_loss = self.denoise_loss_fn(y["denoise"], denoise_out)
        s_loss = self.segment_loss_fn(y["segment"], seg_out)
        c_loss = confidence_f1_loss(
            seg_pred=seg_out,
            confidence_pred=confidence_out,
            gt_labels=y["segment"],
            confidence_weight=self.confidence_weight,
        )
        total = self.denoise_weight * d_loss + s_loss + c_loss

        # F1 and confidence summary values (for logging)
        f1_vals = tf.stop_gradient(compute_track_f1(seg_out, y["segment"]))

        return total, d_loss, s_loss, c_loss, f1_vals, confidence_out

    def _update_trackers(self, total, d_loss, s_loss, c_loss, f1_vals, conf_out):
        self._loss_tracker.update_state(total)
        self._denoise_loss_tracker.update_state(d_loss)
        self._segment_loss_tracker.update_state(s_loss)
        self._confidence_loss_tracker.update_state(c_loss)
        self._mean_f1_tracker.update_state(tf.reduce_mean(f1_vals))
        self._mean_conf_tracker.update_state(tf.reduce_mean(conf_out))

    # ---- train / test steps ----------------------------------------------

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            total, d_loss, s_loss, c_loss, f1_vals, conf_out = self._compute_losses(
                y, outputs
            )

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._update_trackers(total, d_loss, s_loss, c_loss, f1_vals, conf_out)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        outputs = self(x, training=False)
        total, d_loss, s_loss, c_loss, f1_vals, conf_out = self._compute_losses(
            y, outputs
        )
        self._update_trackers(total, d_loss, s_loss, c_loss, f1_vals, conf_out)

        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_and_prepare(
    root_file: str,
    confidence_mode: str,
) -> tuple:
    """Load a ROOT file and prepare training targets.

    Returns ``(X, X_clean, y_seg, confidence)`` where *confidence* is
    ``None`` when not needed.
    """
    need_conf = confidence_mode == CONFIDENCE_EVENT_LEVEL

    if need_conf:
        X, X_clean, y_mup, y_mum, conf = load_data_denoise(
            root_file, return_confidence=True
        )
        if X is None:
            return None, None, None, None
    else:
        result = load_data_denoise(root_file)
        X, X_clean, y_mup, y_mum = result
        conf = None
        if X is None:
            return None, None, None, None

    y_seg = np.stack([y_mup, y_mum], axis=1)  # (N, 2, 62)
    return X, X_clean, y_seg, conf


def _make_targets(X_clean, y_seg, confidence, confidence_mode):
    """Build the ``y`` dict expected by ``model.fit``."""
    targets = {
        "denoise": X_clean,
        "segment": y_seg,
    }
    if confidence_mode == CONFIDENCE_EVENT_LEVEL:
        targets["confidence"] = confidence
    # For track_quality the confidence target is computed dynamically inside
    # TrackFinderWithConfidence.train_step – we pass the segment labels which
    # are already in the dict.
    return targets


def train_model(args: argparse.Namespace) -> None:
    """Train the joint denoising + segmentation (+ confidence) model.

    Supports curriculum learning with low, medium, and high complexity
    datasets.  Uses ``MirroredStrategy`` for multi-GPU training.

    The *confidence_mode* argument controls whether a confidence head is
    attached and how it is trained:

    * ``none``          – legacy two-head model.
    * ``event_level``   – Proposal A (binary presence label, standard BCE).
    * ``track_quality`` – Proposal B (dynamic F1 target, custom train step).
    """

    confidence_mode = args.confidence_mode
    if confidence_mode not in _VALID_CONFIDENCE_MODES:
        raise ValueError(
            f"Invalid --confidence_mode={confidence_mode!r}.  "
            f"Must be one of {sorted(_VALID_CONFIDENCE_MODES)}."
        )

    # ---- Distributed Training ----
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # ---- Load low-complexity training data and validation data ----
    X_train_low, X_clean_train_low, y_train_low, conf_train_low = _load_and_prepare(
        args.train_root_file_low, confidence_mode
    )
    if X_train_low is None:
        return

    X_val, X_clean_val, y_val, conf_val = _load_and_prepare(
        args.val_root_file, confidence_mode
    )
    if X_val is None:
        return

    train_targets_low = _make_targets(
        X_clean_train_low, y_train_low, conf_train_low, confidence_mode
    )
    val_targets = _make_targets(X_clean_val, y_val, conf_val, confidence_mode)

    # ---- Build & compile model ----
    with strategy.scope():
        functional_model = build_model(
            num_detectors=NUM_DETECTORS,
            num_elementIDs=NUM_ELEMENT_IDS,
            use_bn=args.batch_norm,
            dropout_bn=args.dropout_bn,
            dropout_enc=args.dropout_enc,
            denoise_base=args.denoise_base,
            base=args.base,
            use_attn=args.use_attn,
            use_attn_ffn=args.use_attn_ffn,
            dropout_attn=args.dropout_attn,
            confidence_mode=confidence_mode,
        )

        denoise_loss_fn = weighted_bce(pos_weight=args.pos_weight)
        segment_loss_fn = custom_loss

        if confidence_mode == CONFIDENCE_TRACK_QUALITY:
            # Proposal B – wrap functional model in custom class
            model = TrackFinderWithConfidence(
                inputs=functional_model.input,
                outputs=functional_model.outputs,
                denoise_loss_fn=denoise_loss_fn,
                segment_loss_fn=segment_loss_fn,
                confidence_weight=args.confidence_weight,
                denoise_weight=10.0,
            )
            model.summary()

            optimizer = AdamW(
                learning_rate=args.lr_low,
                weight_decay=args.weight_decay,
                clipnorm=args.clipnorm,
            )
            # Only need the optimizer; losses are handled in train_step.
            model.compile(optimizer=optimizer)

        else:
            # confidence_mode is "none" or "event_level" (Proposal A)
            model = functional_model
            model.summary()

            optimizer = AdamW(
                learning_rate=args.lr_low,
                weight_decay=args.weight_decay,
                clipnorm=args.clipnorm,
            )

            losses = {
                "denoise": denoise_loss_fn,
                "segment": segment_loss_fn,
            }
            loss_weights = {
                "denoise": 10.0,
                "segment": 1.0,
            }
            metrics_dict = {
                "denoise": [Precision(name="precision"), Recall(name="recall")],
                "segment": ["accuracy"],
            }

            if confidence_mode == CONFIDENCE_EVENT_LEVEL:
                losses["confidence"] = confidence_bce(
                    confidence_weight=1.0,  # actual weighting via loss_weights
                    pos_weight=args.confidence_pos_weight,
                )
                loss_weights["confidence"] = args.confidence_weight
                metrics_dict["confidence"] = [
                    tf.keras.metrics.BinaryAccuracy(name="acc"),
                    Precision(name="precision"),
                    Recall(name="recall"),
                ]

            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
                metrics=metrics_dict,
            )

    # ---- Training loop (supports curriculum learning) ----

    if args.train_root_file_med and args.train_root_file_high:
        print("Curriculum learning enabled.")

        epochs_low = int(args.epochs * args.low_ratio)
        epochs_med = int(args.epochs * args.med_ratio)
        epochs_high = args.epochs

        # --- Stage 1: low complexity ---
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_low,
            train_targets_low,
            initial_epoch=0,
            epochs=epochs_low,
            batch_size=args.batch_size,
            validation_data=(X_val, val_targets),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del (
            X_train_low,
            X_clean_train_low,
            y_train_low,
            conf_train_low,
            train_targets_low,
        )
        gc.collect()

        # --- Stage 2: medium complexity ---
        X_train_med, X_clean_train_med, y_train_med, conf_train_med = _load_and_prepare(
            args.train_root_file_med, confidence_mode
        )
        if X_train_med is None:
            return
        train_targets_med = _make_targets(
            X_clean_train_med, y_train_med, conf_train_med, confidence_mode
        )

        K.set_value(model.optimizer.learning_rate, args.lr_med)
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_med,
            train_targets_med,
            initial_epoch=epochs_low,
            epochs=epochs_med,
            batch_size=args.batch_size,
            validation_data=(X_val, val_targets),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del (
            X_train_med,
            X_clean_train_med,
            y_train_med,
            conf_train_med,
            train_targets_med,
        )
        gc.collect()

        # --- Stage 3: high complexity ---
        X_train_high, X_clean_train_high, y_train_high, conf_train_high = (
            _load_and_prepare(args.train_root_file_high, confidence_mode)
        )
        if X_train_high is None:
            return
        train_targets_high = _make_targets(
            X_clean_train_high, y_train_high, conf_train_high, confidence_mode
        )

        K.set_value(model.optimizer.learning_rate, args.lr_high)
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        )
        model.fit(
            X_train_high,
            train_targets_high,
            initial_epoch=epochs_med,
            epochs=epochs_high,
            batch_size=args.batch_size,
            validation_data=(X_val, val_targets),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del (
            X_train_high,
            X_clean_train_high,
            y_train_high,
            conf_train_high,
            train_targets_high,
        )
        gc.collect()

    else:
        # Standard training without curriculum learning
        print("Standard training without curriculum learning.")

        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_low,
            train_targets_low,
            initial_epoch=0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, val_targets),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )

    # ---- Save ----
    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow model to predict hit arrays from event hits."
    )

    # ---- data ----
    parser.add_argument(
        "train_root_file_low", type=str, help="Path to the train ROOT file."
    )
    parser.add_argument(
        "val_root_file", type=str, help="Path to the validation ROOT file."
    )
    parser.add_argument(
        "--train_root_file_med",
        type=str,
        default=None,
        help="Train ROOT file with medium complexity for curriculum learning.",
    )
    parser.add_argument(
        "--train_root_file_high",
        type=str,
        default=None,
        help="Train ROOT file with high complexity for curriculum learning.",
    )

    # ---- output ----
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/track_finder_joint.keras",
        help="Path to save the trained model.",
    )

    # ---- optimiser / lr ----
    parser.add_argument(
        "--lr_low",
        type=float,
        default=0.0003,
        help="Learning rate for low complexity data.",
    )
    parser.add_argument(
        "--lr_med",
        type=float,
        default=0.0001,
        help="Learning rate for medium complexity data.",
    )
    parser.add_argument(
        "--lr_high",
        type=float,
        default=0.00003,
        help="Learning rate for high complexity data.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.3,
        help="Factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--patience", type=int, default=12, help="Patience for EarlyStopping."
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=4,
        help="Patience for learning rate scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=1.0,
        help="Hyperparameter for gradient clipping in AdamW.",
    )

    # ---- architecture ----
    parser.add_argument(
        "--batch_norm",
        type=int,
        default=0,
        help="Flag to set batch normalization: [0 = False, 1 = True].",
    )
    parser.add_argument(
        "--use_attn",
        type=int,
        default=0,
        help="Flag to set attention mechanism: [0 = False, 1 = True].",
    )
    parser.add_argument(
        "--use_attn_ffn",
        type=int,
        default=1,
        help="Flag to set feed-forward layers in attention: [0 = False, 1 = True].",
    )
    parser.add_argument(
        "--dropout_bn",
        type=float,
        default=0.0,
        help="Dropout rate for bottleneck layer.",
    )
    parser.add_argument(
        "--dropout_enc",
        type=float,
        default=0.0,
        help="Dropout rate for encoder blocks.",
    )
    parser.add_argument(
        "--dropout_attn",
        type=float,
        default=0.0,
        help="Dropout rate for attention block.",
    )
    parser.add_argument(
        "--denoise_base",
        type=int,
        default=64,
        help="Number of base channels in denoising U-Net++.",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=64,
        help="Number of base channels in segmentation U-Net++.",
    )

    # ---- training ----
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs in training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for mini-batch gradient descent.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive class weight for weighted BCE (denoiser).",
    )
    parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.5,
        help="Fraction of epochs for low complexity data.",
    )
    parser.add_argument(
        "--med_ratio",
        type=float,
        default=0.8,
        help="Fraction of epochs for medium complexity data.",
    )

    # ---- confidence head ----
    parser.add_argument(
        "--confidence_mode",
        type=str,
        default=CONFIDENCE_NONE,
        choices=sorted(_VALID_CONFIDENCE_MODES),
        help=(
            "Confidence head mode.  'none' = no confidence head (legacy).  "
            "'event_level' = Proposal A (binary track presence, BCE loss).  "
            "'track_quality' = Proposal B (F1-overlap target, custom train step)."
        ),
    )
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.1,
        help=(
            "Weight for the confidence loss relative to the segmentation loss.  "
            "Start conservatively (0.05 – 0.2) to avoid degrading the "
            "reconstruction quality of the segmentation head."
        ),
    )
    parser.add_argument(
        "--confidence_pos_weight",
        type=float,
        default=2.0,
        help=(
            "Positive-class up-weighting for the event_level (Proposal A) "
            "confidence BCE loss.  Values > 1 bias the model towards recall "
            "(fewer missed tracks)."
        ),
    )

    args = parser.parse_args()
    train_model(args)
