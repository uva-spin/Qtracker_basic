"""Denoiser U-Net++ based track-counter: classifies number of dimuon pairs (0..max_pairs)"""

# ruff: noqa: E402
import argparse
import gc
import os

os.environ["TF_GPU_ALLOCATOR"] = (
    "cuda_malloc_async"  # Enable asynchronous GPU memory allocation for better performance
)

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import classification_report, confusion_matrix

from backbones import unetpp_backbone
from data_loader import load_data_counter

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Set mixed precision policy for better performance
mixed_precision.set_global_policy("mixed_float16")

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def build_model(
    num_detectors: int = 62,
    num_elementIDs: int = 201,
    use_bn: bool = False,
    dropout_bn: float = 0.0,
    dropout_enc: float = 0.0,
    denoise_base: int = 64,
    max_pairs: int = 3,
) -> tf.keras.Model:
    """
    Build the track-counter model using a single denoiser U-Net++ backbone followed
    by a global pooling head that predicts how many dimuon pairs are in an event.

    Args:
        num_detectors (int): Number of detectors (default: 62).
        num_elementIDs (int): Number of element IDs (default: 201).
        use_bn (bool): Whether to use batch normalization (default: False).
        dropout_bn (float): Dropout rate for bottleneck layer (default: 0.0).
        dropout_enc (float): Dropout rate for encoder blocks (default: 0.0).
        denoise_base (int): Number of base channels in U-Net++ backbone (default: 64).
        max_pairs (int): Maximum number of dimuon pairs; output has max_pairs+1 classes
            corresponding to counts {0, 1, ..., max_pairs} (default: 3).

    Returns:
        tf.keras.Model: The constructed track-counter model.
    """

    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Denoiser backbone (attention disabled — counter does not need axial attention)
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

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    count_output = layers.Dense(
        max_pairs + 1, activation="softmax", name="count", dtype=tf.float32
    )(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[count_output])
    return model


def train_model(args: argparse.Namespace) -> None:
    """
    Train the track-counter model using the provided arguments.
    Supports curriculum learning with low, medium, and high complexity datasets.
    Utilizes MirroredStrategy for multi-GPU distributed training.

    Args:
        args (argparse.Namespace): Command-line arguments for training configuration.
    """

    # Distributed training
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load low complexity training data and validation data
    X_train_low, counts_train_low = load_data_counter(
        args.train_root_file_low, args.max_pairs
    )
    if X_train_low is None:
        return

    X_val, counts_val = load_data_counter(args.val_root_file, args.max_pairs)
    if X_val is None:
        return

    with strategy.scope():
        model = build_model(
            num_detectors=NUM_DETECTORS,
            num_elementIDs=NUM_ELEMENT_IDS,
            use_bn=args.batch_norm,
            dropout_bn=args.dropout_bn,
            dropout_enc=args.dropout_enc,
            denoise_base=args.denoise_base,
            max_pairs=args.max_pairs,
        )
        model.summary()

        optimizer = AdamW(
            learning_rate=args.lr_low,
            weight_decay=args.weight_decay,
            clipnorm=args.clipnorm,
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    if args.train_root_file_med and args.train_root_file_high:
        # Curriculum learning: low → med → high complexity
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
        history = model.fit(
            X_train_low,
            counts_train_low,
            initial_epoch=0,
            epochs=epochs_low,
            batch_size=args.batch_size,
            validation_data=(X_val, counts_val),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        print("Stage 1 (low) history:", history.history)
        del X_train_low, counts_train_low
        gc.collect()

        # --- Stage 2: medium complexity ---
        X_train_med, counts_train_med = load_data_counter(
            args.train_root_file_med, args.max_pairs
        )
        if X_train_med is None:
            return

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
        history = model.fit(
            X_train_med,
            counts_train_med,
            initial_epoch=epochs_low,
            epochs=epochs_med,
            batch_size=args.batch_size,
            validation_data=(X_val, counts_val),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        print("Stage 2 (med) history:", history.history)
        del X_train_med, counts_train_med
        gc.collect()

        # --- Stage 3: high complexity ---
        X_train_high, counts_train_high = load_data_counter(
            args.train_root_file_high, args.max_pairs
        )
        if X_train_high is None:
            return

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
        history = model.fit(
            X_train_high,
            counts_train_high,
            initial_epoch=epochs_med,
            epochs=epochs_high,
            batch_size=args.batch_size,
            validation_data=(X_val, counts_val),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        print("Stage 3 (high) history:", history.history)
        del X_train_high, counts_train_high
        gc.collect()

    else:
        # Standard single-stage training
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
        history = model.fit(
            X_train_low,
            counts_train_low,
            initial_epoch=0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, counts_val),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        print("Training history:", history.history)

    # Evaluation on validation set
    val_preds = model.predict(X_val, batch_size=args.batch_size)
    val_pred_counts = np.argmax(val_preds, axis=-1)
    print("Confusion Matrix:")
    print(confusion_matrix(counts_val, val_pred_counts))
    print("\nClassification Report:")
    print(classification_report(counts_val, val_pred_counts, zero_division=0))

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow model to count the number of dimuon pairs in an event."
    )
    parser.add_argument(
        "train_root_file_low",
        type=str,
        help="Path to the low-complexity train ROOT file.",
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
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/track_counter.keras",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=3,
        help="Maximum number of pairs (counter predicts 0..max_pairs).",
    )
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
        "--batch_norm",
        type=int,
        default=0,
        help="Flag to set batch normalization: [0 = False, 1 = True].",
    )
    parser.add_argument(
        "--use_attn",
        type=int,
        default=0,
        help="Flag kept for CLI parity with TrackFinder (ignored; counter always uses use_attn=False).",
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
        "--denoise_base",
        type=int,
        default=64,
        help="Number of base channels in U-Net++ backbone.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for mini-batch gradient descent.",
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
        help="Gradient clipping norm for AdamW optimizer.",
    )
    parser.add_argument(
        "--low_ratio",
        type=float,
        default=0.5,
        help="Fraction of total epochs to use for low complexity stage.",
    )
    parser.add_argument(
        "--med_ratio",
        type=float,
        default=0.8,
        help="Fraction of total epochs to use for medium complexity stage.",
    )
    args = parser.parse_args()

    # batch_norm and use_attn are stored as ints from argparse; convert to bool
    args.batch_norm = bool(args.batch_norm)

    train_model(args)
