""" Vanilla U-Net++ (no deep supervision or curriculum learning) """

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
# from tensorflow.keras import mixed_precision

from data_loader import load_data
from losses import custom_loss

# mixed_precision.set_global_policy("mixed_float16")

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def conv_block(x, filters, l2=1e-4, use_bn=False, dropout_bn=0.0, dropout=0.0):
    shortcut = x

    # First Conv Layer + Activation
    x = layers.Conv2D(
        filters, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.l2(l2)
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Dropout for bottleneck layers
    if dropout_bn > 0:
        x = layers.Dropout(dropout_bn)(x)

    # Second Conv Layer
    x = layers.Conv2D(
        filters, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.l2(l2)
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)

    # Project shortcut if needed
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    
    x = layers.Activation('relu')(x)

    # Dropout for encoder blocks
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def upsample(x):
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    return x


def build_model(
    num_detectors=62,
    num_elementIDs=201,
    use_bn=False,
    dropout_bn=0.0,
    dropout_enc=0.0,
    base=64,
):
    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    filters = [base, base*2, base*4, base*8, base*16]
    num_pool = 2**(len(filters) - 1)  # 2 ^ n, n = number of max pooling
    closest_even_det = num_pool * math.ceil(num_detectors / num_pool)
    closest_even_elem = num_pool * math.ceil(num_elementIDs / num_pool)
    det_diff = closest_even_det - num_detectors
    elem_diff = closest_even_elem - num_elementIDs
    padding = (
        (det_diff // 2, det_diff - det_diff // 2),
        (elem_diff // 2, elem_diff - elem_diff // 2),
    )

    x = layers.ZeroPadding2D(padding=padding)(input_layer)

    # Encoder (column j=0)
    X = [[None] * len(filters) for _ in range(len(filters))]    # X[i][j]

    X[0][0] = conv_block(x, filters[0], use_bn=use_bn)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(X[0][0])

    X[1][0] = conv_block(pool1, filters[1], use_bn=use_bn, dropout=dropout_enc / 2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(X[1][0])

    X[2][0] = conv_block(pool2, filters[2], use_bn=use_bn, dropout=dropout_enc)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(X[2][0])

    X[3][0] = conv_block(pool3, filters[3], use_bn=use_bn, dropout=dropout_enc)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(X[3][0])

    X[4][0] = conv_block(pool4, filters[4], use_bn=use_bn, dropout_bn=dropout_bn)

    # Column j=1
    for j in range(1, len(filters)):
        for i in range(0, len(filters) - j):
            concat_parts = [X[i][k] for k in range(j)] + [upsample(X[i+1][j-1])]
            X[i][j] = conv_block(layers.Concatenate()(concat_parts), filters[i], use_bn=use_bn)

    # Logits
    x = layers.Cropping2D(cropping=padding)(X[0][j])
    x = layers.Conv2D(2, kernel_size=1)(x)

    # Softmax + permute to match shape required by loss: (batch, 2, det, elem)
    x = layers.Softmax(axis=2)(x)  # softmax over elementID
    output = layers.Permute((3, 1, 2))(x)

    # Initialize model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def train_model(args):
    X_train, y_muPlus_train, y_muMinus_train = load_data(args.train_root_file)
    if X_train is None:
        return
    y_train = np.stack(
        [y_muPlus_train, y_muMinus_train], axis=1
    )  # Shape: (num_events, 2, 62)

    X_val, y_muPlus_val, y_muMinus_val = load_data(args.val_root_file)
    if X_val is None:
        return
    y_val = np.stack(
        [y_muPlus_val, y_muMinus_val], axis=1
    )  # Shape: (num_events, 2, 62)

    if args.denoise_model_path:
        denoise_model = tf.keras.models.load_model(
            args.denoise_model_path, compile=False
        )
        X_train = denoise_model.predict(X_train)
        X_train = (X_train > 0.5).astype(np.float32)  # Binarize
        X_val = denoise_model.predict(X_val)
        X_val = (X_val > 0.5).astype(np.float32)  # Binarize

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=args.patience // 3, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    model = build_model(
        num_detectors=NUM_DETECTORS,
        num_elementIDs=NUM_ELEMENT_IDS, 
        use_bn=args.batch_norm, 
        dropout_bn=args.dropout_bn, 
        dropout_enc=args.dropout_enc, 
        base=args.base,
    )
    model.summary()

    optimizer = AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clipnorm=args.clipnorm,
    )
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler, early_stopping],
    )

    # Plot train and val loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Val Loss Over Epochs")

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "losses.png"))
    plt.show()

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow model to predict hit arrays from event hits."
    )
    parser.add_argument(
        "train_root_file", type=str, help="Path to the train ROOT file."
    )
    parser.add_argument(
        "val_root_file", type=str, help="Path to the validation ROOT file."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/track_finder_unetpp.h5",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00005,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for EarlyStopping."
    )
    parser.add_argument(
        "--batch_norm",
        type=int,
        default=0,
        help="Flag to set batch normalization: [0 = False, 1 = True].",
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
        "--denoise_model_path",
        type=str,
        default=None,
        help="Path to the saved denoiser model file (.h5 or .keras).",
    )
    args = parser.parse_args()

    train_model(args)
