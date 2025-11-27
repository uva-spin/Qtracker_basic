""" U-Net based denoiser + segmenter: end-to-end training """

import argparse
import gc
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall

from backbones import unetpp_backbone
from data_loader import load_data_denoise
from losses import custom_loss, weighted_bce

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


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
) -> tf.keras.Model:
    """
    This function builds the joint denoising and segmentation model using two U-Net++ backbones.
    It first denoises the input hit array and then segments the denoised output end-to-end.

    Args:
        num_detectors (int): Number of detectors (default: 62).
        num_elementIDs (int): Number of element IDs (default: 201).
        use_bn (bool): Whether to use batch normalization (default: False).
        dropout_bn (float): Dropout rate for bottleneck layer (default: 0.0).
        dropout_enc (float): Dropout rate for encoder blocks (default: 0.0).
        denoise_base (int): Number of base channels in U-Net++ for denoising (default: 64).
        base (int): Number of base channels in U-Net++ for segmentation (default: 64).
        use_attn (bool): Whether to use axial attention mechanism in segmentation U-Net++ (default: False).
        use_attn_ffn (bool): Whether to use feed-forward layers in attention (default: True).
        dropout_attn (float): Dropout rate for attention block (default: 0.0).
    
    Returns:
        tf.keras.Model: The constructed joint denoising and segmentation model.
    """

    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Denoising Backbone - first U-Net++
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

    # Denoise Head
    denoise_out = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="denoise")(x)

    # Segmentation Backbone - second U-Net++
    x = unetpp_backbone(
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

    # Segmentation Head
    x = layers.Conv2D(2, kernel_size=1)(x)
    x = layers.Softmax(axis=2)(x)  # softmax over elementID
    seg_output = layers.Permute((3, 1, 2), name="segment")(x)   # (batch, 2, det, elem)

    # Initialize model
    model = tf.keras.Model(inputs=input_layer, outputs=[denoise_out, seg_output])
    return model


def train_model(args: argparse.Namespace) -> None:
    """
    This function trains the joint denoising and segmentation model using the provided arguments.
    It supports curriculum learning with low, medium, and high complexity datasets.
    Additionally, it utilizes distributed training with MirroredStrategy to enable multi-GPU training.

    Args:
        args (argparse.Namespace): Command-line arguments for training configuration.
    """

    # Distributed Training
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load low complexity training data and validation data
    X_train_low, X_clean_train_low, y_muPlus_train_low, y_muMinus_train_low = load_data_denoise(
        args.train_root_file_low
    )
    if X_train_low is None or X_clean_train_low is None:
        return
    y_train_low = np.stack(
        [y_muPlus_train_low, y_muMinus_train_low], axis=1
    )  # Shape: (num_events, 2, 62)

    X_val, X_clean_val, y_muPlus_val, y_muMinus_val = load_data_denoise(
        args.val_root_file
    )
    if X_val is None or X_clean_val is None:
        return
    y_val = np.stack(
        [y_muPlus_val, y_muMinus_val], axis=1
    )  # Shape: (num_events, 2, 62)

    with strategy.scope():
        model = build_model(
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
        )
        model.summary()

        optimizer = AdamW(
            learning_rate=args.lr_low,
            weight_decay=args.weight_decay,
            clipnorm=args.clipnorm,
        )

        """
        Compile the model with multiple losses and metrics
        - Denoising: Weighted BCE to heavily penalize false negatives
        - Segmentation: Custom cross entropy based loss function
        """
        model.compile(
            optimizer=optimizer,
            loss={
                "denoise": weighted_bce(pos_weight=args.pos_weight),
                "segment": custom_loss,
            },
            loss_weights={
                "denoise": 10.0,
                "segment": 1.0,
            },
            metrics={
                "denoise": [Precision(name='precision'), Recall(name='recall')],
                "segment": ["accuracy"],
            }
        )

    if args.train_root_file_med and args.train_root_file_high:     # enable curriculum learning
        print("Curriculum learning enabled.")

        """
        Typical curriculum learning with 3 stages: low, medium, high complexity
        - We determine epochs for each stage based on provided ratios
        - Learning rate is adjusted for each stage
        - Delete training data after each stage to free up memory
        """

        epochs_low = int(args.epochs * args.low_ratio)
        epochs_med = int(args.epochs * args.med_ratio)
        epochs_high = args.epochs

        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_low,
            {"denoise": X_clean_train_low, "segment": y_train_low},
            initial_epoch=0,
            epochs=epochs_low,
            batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del X_train_low, X_clean_train_low, y_train_low
        gc.collect()  

        X_train_med, X_clean_train_med, y_muPlus_train_med, y_muMinus_train_med = load_data_denoise(
            args.train_root_file_med
        )
        if X_train_med is None or X_clean_train_med is None:
            return
        y_train_med = np.stack(
            [y_muPlus_train_med, y_muMinus_train_med], axis=1
        )  # Shape: (num_events, 2, 62)

        K.set_value(model.optimizer.learning_rate, args.lr_med)
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_med,
            {"denoise": X_clean_train_med, "segment": y_train_med},
            initial_epoch=epochs_low,
            epochs=epochs_med,
            batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del X_train_med, X_clean_train_med, y_train_med
        gc.collect()

        X_train_high, X_clean_train_high, y_muPlus_train_high, y_muMinus_train_high = load_data_denoise(
            args.train_root_file_high
        )
        if X_train_high is None or X_clean_train_high is None:
            return
        y_train_high = np.stack(
            [y_muPlus_train_high, y_muMinus_train_high], axis=1
        )  # Shape: (num_events, 2, 62)

        K.set_value(model.optimizer.learning_rate, args.lr_high)
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        )

        model.fit(
            X_train_high,
            {"denoise": X_clean_train_high, "segment": y_train_high},
            initial_epoch=epochs_med,
            epochs=epochs_high,
            batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )
        del X_train_high, X_clean_train_high, y_train_high
        gc.collect()

    else:   # standard training without curriculum learning
        print("Standard training without curriculum learning.")

        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6
        )
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=False
        )
        model.fit(
            X_train_low,
            {"denoise": X_clean_train_low, "segment": y_train_low},
            initial_epoch=0,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=[lr_scheduler, early_stopping],
            verbose=2,
        )

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow model to predict hit arrays from event hits."
    )
    parser.add_argument(
        "train_root_file_low", type=str, help="Path to the train ROOT file."
    )
    parser.add_argument(
        "val_root_file", type=str, help="Path to the validation ROOT file."
    )
    parser.add_argument(
        "--train_root_file_med", type=str, default=None, help="Train ROOT file with medium complexity for curriculum learning."
    )
    parser.add_argument(
        "--train_root_file_high", type=str, default=None, help="Train ROOT file with high complexity for curriculum learning."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/track_finder_joint.keras",
        help="Path to save the trained model.",
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
        "--lr_patience", type=int, default=4, help="Patience for learning rate scheduler."
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
        help="Number of base channels in U-Net++.",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=64,
        help="Number of base channels in U-Net++.",
    )
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
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive class weight for weighted BCE.",
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
    args = parser.parse_args()

    train_model(args)
