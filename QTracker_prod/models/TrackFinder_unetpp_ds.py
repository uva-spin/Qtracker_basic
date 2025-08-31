"""Custom U-Net for particle track reconstruction"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
import tensorflow as tf
from tensorflow.keras import layers, regularizers, mixed_precision
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)


def load_data(root_file):
    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")

    if not tree:
        print("Error: Tree not found in file.")
        return None, None, None

    num_detectors = 62
    num_elementIDs = 201

    X = []
    y_muPlus = []
    y_muMinus = []

    for event in tree:
        event_hits_matrix = np.zeros((num_detectors, num_elementIDs), dtype=np.float32)

        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                event_hits_matrix[det_id, elem_id] = 1

        mu_plus_array = np.array(event.HitArray_mup, dtype=np.int32)
        mu_minus_array = np.array(event.HitArray_mum, dtype=np.int32)

        X.append(event_hits_matrix)
        y_muPlus.append(mu_plus_array)
        y_muMinus.append(mu_minus_array)

    X = np.array(X)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    y_muPlus = np.array(y_muPlus)
    y_muMinus = np.array(y_muMinus)

    return X, y_muPlus, y_muMinus


def custom_loss(y_true, y_pred, focal=False):
    def sparse_focal_crossentropy(y_true, y_pred, gamma=2.0, alpha=1.0):
        """
        y_true: (B, Det) int indices
        y_pred: (B, Det, Elem) probabilities (softmax), float
        returns: (B, Det) focal CE per detector
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # p_t = prob of the true class
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=tf.float32)  # (B,Det,Elem)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)                              # (B,Det)
        p_t = tf.clip_by_value(p_t, 1e-9, 1.0)

        focal = -alpha * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)                  # (B,Det)
        return focal

    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.cast(tf.squeeze(y_muPlus_pred, axis=1), tf.float32)
    y_muMinus_pred = tf.cast(tf.squeeze(y_muMinus_pred, axis=1), tf.float32)

    if focal:
        loss_mup = sparse_focal_crossentropy(y_muPlus_true, y_muPlus_pred)
        loss_mum = sparse_focal_crossentropy(y_muMinus_true, y_muMinus_pred)
    else:
        loss_mup = sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
        loss_mum = sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    overlap_penalty = tf.reduce_sum(y_muPlus_pred * y_muMinus_pred, axis=-1)
    overlap_penalty = tf.reduce_mean(overlap_penalty, axis=-1)

    ce_loss = tf.reduce_mean(loss_mup + loss_mum, axis=-1)
    return tf.reduce_mean(ce_loss + 0.1 * overlap_penalty)


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


def head(feature, padding, name):
    x = layers.Cropping2D(cropping=padding)(feature)
    x = layers.Conv2D(2, kernel_size=1, name=name)(x)
    x = layers.Softmax(axis=2)(x)  # softmax over elementID
    return layers.Permute((3, 1, 2))(
        x
    )  # permute to match shape required by loss: (batch, 2, det, elem)


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
    depth = len(filters) - 1    
    num_pool = 2 ** depth
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
    x = [head(X[0][j], padding, name=f'logits_{j}') for j in range(1, len(filters))]
    output = list(reversed(x))

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

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=args.patience // 3, min_lr=1e-6
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    model = build_model(
        use_bn=args.batch_norm, dropout_bn=args.dropout_bn, 
        dropout_enc=args.dropout_enc, base=args.base
    )
    model.summary()

    optimizer = AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clipnorm=args.clipnorm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    num_outputs = len(model.outputs)
    losses = [
        lambda y_true, y_pred: custom_loss(y_true, y_pred, focal=True)
    ] + [custom_loss] * (num_outputs - 1)
    model.compile(
        optimizer=optimizer, 
        loss=losses,
        loss_weights=[1.0 / (2 ** i) for i in range(num_outputs)],   # exponential decay 
    )

    history = model.fit(
        X_train,
        [y_train] * num_outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, [y_val] * num_outputs),
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

    model.save_weights(args.output_model)
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
        default="checkpoints/track_finder_unetpp_ds.weights.h5",
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
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    args = parser.parse_args()

    train_model(args)
