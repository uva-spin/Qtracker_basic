import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

def load_data(root_file):
    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")
    if not tree:
        print("Error: Tree not found in file.")
        return None, None, None

    num_detectors = 62
    num_elementIDs = 201

    X, y_muPlus, y_muMinus = [], [], []
    for event in tree:
        matrix = np.zeros((num_detectors, num_elementIDs), dtype=np.float32)
        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                matrix[det_id, elem_id] = 1
        X.append(matrix)
        y_muPlus.append(np.array(event.HitArray_mup, dtype=np.int32))
        y_muMinus.append(np.array(event.HitArray_mum, dtype=np.int32))

    X = np.array(X)[..., np.newaxis]
    y_muPlus = np.array(y_muPlus)
    y_muMinus = np.array(y_muMinus)
    return X, y_muPlus, y_muMinus


def custom_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, 2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, 2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)
    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)
    return tf.reduce_mean(loss_mup + loss_mum + 0.1 * overlap_penalty)


def cbam_block(x, reduction_ratio=8):
    # Channel Attention
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    shared_dense = tf.keras.Sequential([
        tf.keras.layers.Dense(x.shape[-1] // reduction_ratio, activation='relu', use_bias=False),
        tf.keras.layers.Dense(x.shape[-1], use_bias=False)
    ])
    ca = tf.keras.layers.Activation('sigmoid')(shared_dense(avg_pool) + shared_dense(max_pool))
    x = tf.keras.layers.Multiply()([x, ca])

    # Spatial Attention
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    sa = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    sa = tf.keras.layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(sa)
    x = tf.keras.layers.Multiply()([x, sa])

    return x


def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                                kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                                kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    x = cbam_block(x)  # Add CBAM after residual merge
    return x


def build_model(num_detectors=62, num_elementIDs=201, learning_rate=0.00005):
    inputs = tf.keras.Input(shape=(num_detectors, num_elementIDs, 1))

    # Front-end VGG-style
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 256)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 512)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 512)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Dense Head
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    x = tf.keras.layers.Dense(2 * num_detectors * num_elementIDs, activation='softmax')(x)
    outputs = tf.keras.layers.Reshape((2, num_detectors, num_elementIDs))(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=custom_loss, metrics=['accuracy'])

    return model


def train_model(root_file, output_model, learning_rate=0.00005):
    X, y_muPlus, y_muMinus = load_data(root_file)
    if X is None:
        return
    y = np.stack([y_muPlus, y_muMinus], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(learning_rate=learning_rate)
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))
    model.save(output_model)
    print(f"Model saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("--output_model", type=str, default="models/track_finder_cbam.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    args = parser.parse_args()

    train_model(args.root_file, args.output_model, args.learning_rate)
