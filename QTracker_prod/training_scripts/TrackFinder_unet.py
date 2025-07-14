import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    Permute,
    Activation,
    BatchNormalization,
    Dropout
)


# Ensure the models directory exists
os.makedirs("models", exist_ok=True)


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


def custom_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    return tf.reduce_mean(loss_mup + loss_mum + 0.1 * overlap_penalty)


def unet_block(x, filters, use_bn=False, dropout_rate=0.0):
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_model(num_detectors=62, num_elementIDs=201, learning_rate=0.00005, use_bn=False, dropout_rate=0.0):
    input_layer = Input(shape=(num_detectors, num_elementIDs, 1))

    # Encoder
    enc1 = unet_block(input_layer, 64, use_bn=use_bn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(enc1)

    enc2 = unet_block(pool1, 128, use_bn=use_bn)
    pool2 = MaxPooling2D(pool_size=(2, 2))(enc2)

    enc3 = unet_block(pool2, 256, use_bn=use_bn)
    pool3 = MaxPooling2D(pool_size=(2, 2))(enc3)

    enc4 = unet_block(pool3, 512, use_bn=use_bn)
    pool4 = MaxPooling2D(pool_size=(2, 2))(enc4)

    # Bottleneck
    enc5 = unet_block(pool4, 1024, use_bn=use_bn, dropout_rate=dropout_rate)

    # Decoder
    dec1 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(enc5) # padding same avoids cropping
    dec1 = concatenate([dec1, enc4])    # skip connections
    dec1 = unet_block(dec1, 512, use_bn=use_bn)

    dec2 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(dec1) # padding same avoids cropping
    dec2 = concatenate([dec2, enc3])    # skip connections
    dec2 = unet_block(dec2, 256, use_bn=use_bn)

    dec3 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(dec2) # padding same avoids cropping
    dec3 = concatenate([dec3, enc2])    # skip connections
    dec3 = unet_block(dec3, 128, use_bn=use_bn)

    dec4 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(dec3) # padding same avoids cropping
    dec4 = concatenate([dec4, enc1])    # skip connections
    dec4 = unet_block(dec4, 64, use_bn=use_bn)

    # Output layer
    x = Conv2D(2, kernel_size=1, activation='softmax')(dec4)
    output = Permute((3, 1, 2))(x)  # permute to match shape required by loss: (batch, 2, det, elem)

    # Initialize model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

    return model


def train_model(root_file, output_model, learning_rate=0.00005, use_bn=False, dropout_rate=0.0):
    X, y_muPlus, y_muMinus = load_data(root_file)

    if X is None:
        return

    y = np.stack([y_muPlus, y_muMinus], axis=1)  # Shape: (num_events, 2, 62)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(learning_rate=learning_rate, use_bn=use_bn, dropout_rate=dropout_rate)
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

    model.save(output_model)
    print(f"Model saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("--output_model", type=str, default="models/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    args = parser.parse_args()

    use_bn = bool(args.batch_norm)

    train_model(
        args.root_file, 
        args.output_model, 
        args.learning_rate, 
        use_bn=use_bn, 
        dropout_rate=args.dropout_rate      # recommend 0.5 as starting point
    )
