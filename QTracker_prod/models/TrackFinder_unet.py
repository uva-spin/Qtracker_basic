""" Custom U-Net for particle track reconstruction """

import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
import math
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW

from data_loader import load_data
from losses import custom_loss

# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)


def conv_block(x, filters, l2=1e-4, use_bn=False, dropout_bn=0.0, dropout_enc=0.0):
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
    x = layers.Activation('relu')(x)

    # Dropout for encoder blocks
    if dropout_enc > 0:
        x = layers.Dropout(dropout_enc)(x)
    return x


def res_conv_block(x, filters, l2=1e-4, use_bn=False, dropout_bn=0.0, dropout_enc=0.0):
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
    if dropout_enc > 0:
        x = layers.Dropout(dropout_enc)(x)
    return x


def build_model(num_detectors=62, num_elementIDs=201, base=64, use_bn=False, dropout_bn=0.0, dropout_enc=0.0, backbone=None):
    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    filters = [base, base*2, base*4, base*8, base*16]
    n = 5 if backbone == 'resnet50' else len(filters) - 1
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
    if backbone == 'resnet50':
        x = layers.Concatenate()([x, x, x])
        # x = Conv2D(3, kernel_size=1, padding='same')(x)
        backbone = ResNet50(include_top=False, input_tensor=x, weights=None)

        # Partially freeze backbone and alter stride for compatibility with U-Net decoder
        # backbone.trainable = False
        for layer in backbone.layers:
            if layer.name == 'conv1_conv':
                layer.strides = (1, 1)
            # if layer.name.startswith('conv5_') or layer.name.startswith('conv4_'):
            #     layer.trainable = True

        enc1 = backbone.get_layer('conv1_relu').output
        enc2 = backbone.get_layer('conv2_block3_out').output
        enc3 = backbone.get_layer('conv3_block4_out').output
        enc4 = backbone.get_layer('conv4_block6_out').output
        enc5 = backbone.get_layer('conv5_block3_out').output
    else:
        enc1 = conv_block(x, filters[0], use_bn=use_bn)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)

        enc2 = conv_block(pool1, filters[1], use_bn=use_bn)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)

        enc3 = conv_block(pool2, filters[2], use_bn=use_bn)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)

        enc4 = conv_block(pool3, filters[3], use_bn=use_bn, dropout_enc=dropout_enc)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)

        # Bottleneck
        enc5 = conv_block(pool4, filters[4], use_bn=use_bn, dropout_bn=dropout_bn)

    # Decoder
    dec1 = layers.Conv2DTranspose(filters[3], kernel_size=3, strides=2, padding='same')(enc5) # padding same avoids cropping
    dec1 = layers.concatenate([dec1, enc4])    # skip connections
    dec1 = conv_block(dec1, filters[3], use_bn=use_bn)

    dec2 = layers.Conv2DTranspose(filters[2], kernel_size=3, strides=2, padding='same')(dec1) # padding same avoids cropping
    dec2 = layers.concatenate([dec2, enc3])    # skip connections
    dec2 = conv_block(dec2, filters[2], use_bn=use_bn)

    dec3 = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding='same')(dec2) # padding same avoids cropping
    dec3 = layers.concatenate([dec3, enc2])    # skip connections
    dec3 = conv_block(dec3, filters[1], use_bn=use_bn)

    dec4 = layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding='same')(dec3)
    dec4 = layers.concatenate([dec4, enc1])    # skip connections
    dec4 = layers.Cropping2D(cropping=padding)(dec4)   # Remove extra padding
    dec4 = conv_block(dec4, filters[0], use_bn=use_bn)

    # Output layer
    x = layers.Conv2D(2, kernel_size=1)(dec4)
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
        use_bn=args.batch_norm, 
        dropout_bn=args.dropout_bn, 
        dropout_enc=args.dropout_enc, 
        backbone=args.backbone,
    )
    model.summary()

    optimizer = AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    model.compile(
        optimizer=optimizer, 
        loss=custom_loss, 
        metrics=['accuracy']
    )
    
    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping])

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
    parser.add_argument("--base", type=int, default=64, help="Base number of channels in U-Net.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--dropout_bn", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    parser.add_argument("--dropout_enc", type=float, default=0.0, help="Dropout rate for encoder blocks.")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone encoder. Available: [None, 'resnet50'].")
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
    args = parser.parse_args()

    train_model(args)
