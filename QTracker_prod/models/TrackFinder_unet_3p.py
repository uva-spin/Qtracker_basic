""" Custom U-Net for particle track reconstruction """

import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
import math
import matplotlib.pyplot as plt
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import AdamW

from data_loader import load_data
from losses import custom_loss_mp

mixed_precision.set_global_policy("mixed_float16")

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)


def conv_block(x, filters, use_bn=False, dropout=0.0):
    shortcut = x

    # First Conv Layer + Activation
    x = layers.Conv2D(
        filters, kernel_size=3, padding='same'
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)

    # Project shortcut if needed
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])

    x = layers.Activation('relu')(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x


def enc_block(x, filters, use_bn=False, dropout=0.0):
    x = conv_block(x, filters, use_bn)
    x = conv_block(x, filters, use_bn, dropout)
    return x


def build_model(
    num_detectors=62, 
    num_elementIDs=201, 
    base=64, 
    use_bn=False,
    dropout_bn=0.0,
    dropout=0.0,
):
    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    filters = [base, base*2, base*4, base*8, base*16]
    n = len(filters) - 1
    num_pool = 2 ** n   # 2 ^ n, n = number of max pooling
    closest_even_det = num_pool * math.ceil(num_detectors / num_pool)
    closest_even_elem = num_pool * math.ceil(num_elementIDs / num_pool)
    det_diff = closest_even_det - num_detectors
    elem_diff = closest_even_elem - num_elementIDs
    padding = (
        (det_diff // 2, det_diff - det_diff // 2),
        (elem_diff // 2, elem_diff - elem_diff // 2)
    )

    x = layers.ZeroPadding2D(padding=padding)(input_layer)

    # Encoder
    enc1 = enc_block(x, filters[0], use_bn=use_bn)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)

    enc2 = enc_block(pool1, filters[1], use_bn=use_bn)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)

    enc3 = enc_block(pool2, filters[2], use_bn=use_bn)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)

    enc4 = enc_block(pool3, filters[3], use_bn=use_bn, dropout=dropout)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)

    # Bottleneck
    enc5 = conv_block(pool4, filters[4], use_bn=use_bn, dropout=dropout_bn)
    enc5 = conv_block(enc5, filters[4], use_bn=use_bn)

    # Decoder 4
    enc1_dec4 = layers.MaxPooling2D((8, 8))(enc1)
    enc1_dec4 = conv_block(enc1_dec4, filters[0], use_bn=use_bn)

    enc2_dec4 = layers.MaxPooling2D((4, 4))(enc2)
    enc2_dec4 = conv_block(enc2_dec4, filters[0], use_bn=use_bn)

    enc3_dec4 = layers.MaxPooling2D((2, 2))(enc3)
    enc3_dec4 = conv_block(enc3_dec4, filters[0], use_bn=use_bn)

    enc4_dec4 = conv_block(enc4, filters[0], use_bn=use_bn)

    enc5_dec4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(enc5)
    enc5_dec4 = conv_block(enc5_dec4, filters[0], use_bn=use_bn)

    dec4 = layers.Concatenate()([enc1_dec4, enc2_dec4, enc3_dec4, enc4_dec4, enc5_dec4])
    dec4 = conv_block(dec4, filters[0] * len(filters), use_bn=use_bn)

    # Decoder 3
    enc1_dec3 = layers.MaxPooling2D((4, 4))(enc1)
    enc1_dec3 = conv_block(enc1_dec3, filters[0], use_bn=use_bn)

    enc2_dec3 = layers.MaxPooling2D((2, 2))(enc2)
    enc2_dec3 = conv_block(enc2_dec3, filters[0], use_bn=use_bn)

    enc3_dec3 = conv_block(enc3, filters[0], use_bn=use_bn)

    dec4_dec3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(dec4)
    dec4_dec3 = conv_block(dec4_dec3, filters[0], use_bn=use_bn)

    enc5_dec3 = layers.UpSampling2D((4, 4), interpolation="bilinear")(enc5)
    enc5_dec3 = conv_block(enc5_dec3, filters[0], use_bn=use_bn)

    dec3 = layers.Concatenate()([enc1_dec3, enc2_dec3, enc3_dec3, dec4_dec3, enc5_dec3])
    dec3 = conv_block(dec3, filters[0] * len(filters), use_bn=use_bn)

    # Decoder 2
    enc1_dec2 = layers.MaxPooling2D((2, 2))(enc1)
    enc1_dec2 = conv_block(enc1_dec2, filters[0], use_bn=use_bn)

    enc2_dec2 = conv_block(enc2, filters[0], use_bn=use_bn)

    dec3_dec2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(dec3)
    dec3_dec2 = conv_block(dec3_dec2, filters[0], use_bn=use_bn)

    dec4_dec2 = layers.UpSampling2D((4, 4), interpolation="bilinear")(dec4)
    dec4_dec2 = conv_block(dec4_dec2, filters[0], use_bn=use_bn)

    enc5_dec2 = layers.UpSampling2D((8, 8), interpolation="bilinear")(enc5)
    enc5_dec2 = conv_block(enc5_dec2, filters[0], use_bn=use_bn)

    dec2 = layers.Concatenate()([enc1_dec2, enc2_dec2, dec3_dec2, dec4_dec2, enc5_dec2])
    dec2 = conv_block(dec2, filters[0] * len(filters), use_bn=use_bn)

    # Decoder 1
    enc1_dec1 = conv_block(enc1, filters[0], use_bn=use_bn)

    dec2_dec1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(dec2)
    dec2_dec1 = conv_block(dec2_dec1, filters[0], use_bn=use_bn)

    dec3_dec1 = layers.UpSampling2D((4, 4), interpolation="bilinear")(dec3)
    dec3_dec1 = conv_block(dec3_dec1, filters[0], use_bn=use_bn)

    dec4_dec1 = layers.UpSampling2D((8, 8), interpolation="bilinear")(dec4)
    dec4_dec1 = conv_block(dec4_dec1, filters[0], use_bn=use_bn)

    enc5_dec1 = layers.UpSampling2D((16, 16), interpolation="bilinear")(enc5)
    enc5_dec1 = conv_block(enc5_dec1, filters[0], use_bn=use_bn)

    dec1 = layers.Concatenate()([enc1_dec1, dec2_dec1, dec3_dec1, dec4_dec1, enc5_dec1])
    dec1 = layers.Cropping2D(cropping=padding)(dec1)   # Remove extra padding
    dec1 = conv_block(dec1, filters[0] * len(filters), use_bn=use_bn)

    # Output layer
    x = layers.Conv2D(2, kernel_size=1)(dec1)
    x = layers.Softmax(axis=2)(x)      # softmax over elementID
    output = layers.Permute((3, 1, 2))(x)  # permute to match shape required by loss: (batch, 2, det, elem)

    # Initialize model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def train_model(args):
    X_train, y_muPlus_train, y_muMinus_train = load_data(args.train_root_file)
    if X_train is None:
        return
    y_train = np.stack([y_muPlus_train, y_muMinus_train], axis=1)  # Shape: (num_events, 2, 62)

    X_val, y_muPlus_val, y_muMinus_val = load_data(args.val_root_file)
    if X_val is None:
        return
    y_val = np.stack([y_muPlus_val, y_muMinus_val], axis=1)  # Shape: (num_events, 2, 62)

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=args.patience // 3,
        min_lr=1e-6
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

    model = build_model(
        base=args.base, 
        use_bn=bool(args.batch_norm), 
        dropout_bn=args.dropout_bn,
        dropout=args.dropout,
    )
    model.summary()

    optimizer = AdamW(
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    model.compile(
        optimizer=optimizer, 
        loss=custom_loss_mp, 
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, 
        y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        validation_data=(X_val, y_val), 
        callbacks=[lr_scheduler, early_stopping]
    )

    # Plot train and val loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Val Loss Over Epochs')

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "losses.png"))
    plt.show()

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("train_root_file", type=str, help="Path to the train ROOT file.")
    parser.add_argument("val_root_file", type=str, help="Path to the validation ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder_unet.h5", help="Path to save the trained model.")

    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for EarlyStopping.")
    parser.add_argument("--dropout_bn", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for encoder blocks.")
    parser.add_argument("--base", type=int, default=64, help="Base number of channels in U-Net.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs in training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for mini-batch gradient descent.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps.")
    args = parser.parse_args()

    train_model(args)
