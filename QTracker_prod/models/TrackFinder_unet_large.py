import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

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

def crop_to_match(skip, up):
    def crop(inputs):
        skip_tensor, up_tensor = inputs
        sh, sw = tf.shape(skip_tensor)[1], tf.shape(skip_tensor)[2]
        uh, uw = tf.shape(up_tensor)[1], tf.shape(up_tensor)[2]
        crop_h = sh - uh
        crop_w = sw - uw
        crop_top = crop_h // 2
        crop_bottom = crop_h - crop_top
        crop_left = crop_w // 2
        crop_right = crop_w - crop_left
        return skip_tensor[:, crop_top:sh - crop_bottom, crop_left:sw - crop_right, :]
    return tf.keras.layers.Lambda(crop)([skip, up])



def build_model(num_detectors=62, num_elementIDs=201, learning_rate=0.00005):
    inputs = tf.keras.Input(shape=(num_detectors, num_elementIDs, 1))  # (62, 201, 1)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    b = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)

    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
    c4_cropped = crop_to_match(c4, u4)
    u4 = tf.keras.layers.concatenate([u4, c4_cropped])

    u3 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u4)
    c3_cropped = crop_to_match(c3, u3)
    u3 = tf.keras.layers.concatenate([u3, c3_cropped])

    u2 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u3)
    c2_cropped = crop_to_match(c2, u2)
    u2 = tf.keras.layers.concatenate([u2, c2_cropped])

    u1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u2)
    c1_cropped = crop_to_match(c1, u1)
    u1 = tf.keras.layers.concatenate([u1, c1_cropped])


    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)

    # Flatten and output per track
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(2 * num_detectors * num_elementIDs, activation='softmax')(x)
    output = tf.keras.layers.Reshape((2, num_detectors, num_elementIDs))(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

    return model


def train_model(root_file, output_model, learning_rate=0.00005):
    X, y_muPlus, y_muMinus = load_data(root_file)

    if X is None:
        return

    y = np.stack([y_muPlus, y_muMinus], axis=1)  # Shape: (num_events, 2, 62)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(learning_rate=learning_rate)
    model.fit(X_train, y_train, epochs=40, batch_size=1, validation_data=(X_test, y_test))

    model.save(output_model)
    print(f"Model saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    args = parser.parse_args()

    train_model(args.root_file, args.output_model, args.learning_rate)
    