import os
import ROOT
import numpy as np
#import tensorflow as tf
import argparse
#from tensorflow.keras import regularizers
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split

def load_data(root_file):
    """Loads detector hits and hit arrays from ROOT file and converts them into a binary hit matrix."""
    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")

    if not tree:
        print("Error: Tree not found in file.")
        return None, None, None

    num_detectors = 62   # Number of detectors
    num_elementIDs = 201 # Maximum element IDs per detector

    X = []  # Detector hit matrices (62 detectors x 201 elementIDs)
    y_muPlus = []  # 62-slot hit array for mu+
    y_muMinus = []  # 62-slot hit array for mu-

    # Loop over events
    for event in tree:
        event_hits_matrix = np.zeros((num_detectors, num_elementIDs), dtype=np.float32)

        # Populate the matrix based on integer hit lists
        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                event_hits_matrix[det_id, elem_id] = 1  # Mark hit

        # Extract actual elementIDs (not binary values)
        mu_plus_array = np.array(event.HitArray_mup, dtype=np.int32)
        mu_minus_array = np.array(event.HitArray_mum, dtype=np.int32)

        X.append(event_hits_matrix)
        y_muPlus.append(mu_plus_array)
        y_muMinus.append(mu_minus_array)

    # Convert lists to NumPy arrays
    X = np.array(X)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    y_muPlus = np.array(y_muPlus)  # Shape: (num_events, 62)
    y_muMinus = np.array(y_muMinus)  # Shape: (num_events, 62)

    return X, y_muPlus, y_muMinus


def custom_loss(y_true, y_pred):
    """
    Custom loss function that penalizes the model if it assigns the same elementID
    for both mu+ and mu- in the same detectorID.
    """
    # y_true is expected to have shape: (batch, 2, 62) -> integer class labels
    # y_pred has shape: (batch, 2, 62, 201) -> probability distribution over 201 elementIDs

    # Split labels and predictions into mu+ and mu-
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    # Remove singleton dimensions
    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)  # Shape: (batch, 62)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)  # Shape: (batch, 62)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)  # Shape: (batch, 62, 201)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)  # Shape: (batch, 62, 201)

    # Compute sparse categorical cross-entropy loss separately for mu+ and mu-
    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    # Penalize overlap between mu+ and mu-
    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    return tf.reduce_mean(loss_mup + loss_mum + 0.1 * overlap_penalty)


def build_model(num_detectors=62, num_elementIDs=201, learning_rate=0.00005):


def train_model(root_file, output_model, learning_rate=0.00005):
    """Trains the TensorFlow model using hits as input and hit arrays as output."""
    X, y_muPlus, y_muMinus = load_data(root_file)

    if X is None:
        return

    # Combine y_muPlus and y_muMinus into a single target array
    y = np.stack([y_muPlus, y_muMinus], axis=1)  # Shape: (num_events, 2, 62)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model with the specified learning rate
    model = build_model(learning_rate=learning_rate)

    # Train model
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

    # Save model
    model.save(output_model)
    print(f"Model saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    args = parser.parse_args()

    train_model(args.root_file, args.output_model, args.learning_rate)
