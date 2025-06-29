import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

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
    """Builds a CNN model that processes individual event-track combinations."""
    # Input layer for a single event-track combination
    inputs = tf.keras.layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # 1st Conv Block
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
                               kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 2nd Conv Block
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same",
                               kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 3rd Conv Block
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same",
                               kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten & Fully Connected Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layers for mu+ and mu-
    mu_plus_output = tf.keras.layers.Dense(num_detectors * num_elementIDs, activation='softmax', name="mu_plus_output")(x)
    mu_minus_output = tf.keras.layers.Dense(num_detectors * num_elementIDs, activation='softmax', name="mu_minus_output")(x)
    
    # Reshape outputs to (num_detectors, num_elementIDs)
    mu_plus_output = tf.keras.layers.Reshape((1,1,num_detectors, num_elementIDs), name="mu_plus_reshaped")(mu_plus_output)
    mu_minus_output = tf.keras.layers.Reshape((1,1,num_detectors, num_elementIDs), name="mu_minus_reshaped")(mu_minus_output)

    # Create model with two outputs
    model = tf.keras.Model(inputs=inputs, outputs=[mu_plus_output, mu_minus_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                  loss={'mu_plus_reshaped': 'sparse_categorical_crossentropy', 
                        'mu_minus_reshaped': 'sparse_categorical_crossentropy'},
                  metrics={'mu_plus_reshaped': 'accuracy', 
                           'mu_minus_reshaped': 'accuracy'})
    return model


def process_events_with_arbitrary_dimensions(model, events_data):
    """
    Process arbitrary number of events with arbitrary tracks per event.
    This function handles the dynamic reshaping of data.
    """
    results_mu_plus = []
    results_mu_minus = []
    
    for event_idx, event in enumerate(events_data):
        event_tracks_mu_plus = []
        event_tracks_mu_minus = []
        
        for track_idx, track in enumerate(event.tracks):
            # Reshape track data to (1, num_detectors, num_elementIDs, 1)
            track_input = track.reshape(1, 62, 201, 1)
            
            # Predict
            mu_plus_pred, mu_minus_pred = model.predict(track_input, verbose=0)
            
            event_tracks_mu_plus.append(mu_plus_pred[0])  # Remove batch dimension
            event_tracks_mu_minus.append(mu_minus_pred[0])
        
        # Convert to numpy arrays with shape (num_tracks, num_detectors, num_elementIDs)
        results_mu_plus.append(np.array(event_tracks_mu_plus))
        results_mu_minus.append(np.array(event_tracks_mu_minus))
    
    return results_mu_plus, results_mu_minus


def plot_training_history(history, output_model):
    """Plot and save training history curves."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot 1: Total Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Mu+ Loss
    axes[0, 1].plot(history.history['mu_plus_reshaped_loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_mu_plus_reshaped_loss'], label='Validation Loss')
    axes[0, 1].set_title('Mu+ Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Mu- Loss
    axes[1, 0].plot(history.history['mu_minus_reshaped_loss'], label='Training Loss')
    axes[1, 0].plot(history.history['val_mu_minus_reshaped_loss'], label='Validation Loss')
    axes[1, 0].set_title('Mu- Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Accuracy
    axes[1, 1].plot(history.history['mu_plus_reshaped_accuracy'], label='Mu+ Training Accuracy')
    axes[1, 1].plot(history.history['val_mu_plus_reshaped_accuracy'], label='Mu+ Validation Accuracy')
    axes[1, 1].plot(history.history['mu_minus_reshaped_accuracy'], label='Mu- Training Accuracy')
    axes[1, 1].plot(history.history['val_mu_minus_reshaped_accuracy'], label='Mu- Validation Accuracy')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_model.replace('.keras', '_training_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_file}")
    
    # Also save as PDF for vector graphics
    pdf_file = output_model.replace('.keras', '_training_curves.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Training curves PDF saved to {pdf_file}")
    
    plt.show()


def train_model(root_file, output_model, learning_rate=0.00005):
    """Trains the TensorFlow model using hits as input and hit arrays as output."""
    X, y_muPlus, y_muMinus = load_data(root_file)

    if X is None:
        return

    # Train/test split - keep mu+ and mu- separate for dual output model
    X_train, X_test, y_muPlus_train, y_muPlus_test, y_muMinus_train, y_muMinus_test = train_test_split(
        X, y_muPlus, y_muMinus, test_size=0.2, random_state=42
    )

    # Build model with the specified learning rate
    model = build_model(learning_rate=learning_rate)

    # Train model with separate targets for each output
    history = model.fit(X_train, 
                        {'mu_plus_reshaped': y_muPlus_train, 'mu_minus_reshaped': y_muMinus_train}, 
                        epochs=40, 
                        batch_size=32, 
                        validation_data=(X_test, {'mu_plus_reshaped': y_muPlus_test, 'mu_minus_reshaped': y_muMinus_test}))

    # Save model
    model.save(output_model)
    print(f"Model saved to {output_model}")
    
    # Save training history
    # Convert history to serializable format
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]
    
    # Save to JSON file
    history_file = output_model.replace('.keras', '_history.json')
    with open(history_file, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {history_file}")
    
    # Plot and save training curves
    plot_training_history(history, output_model)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    for key, values in history.history.items():
        if 'loss' in key:
            print(f"{key}: {values[-1]:.4f}")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("--output_model", type=str, default="models/track_finder.keras", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)

    train_model(args.root_file, args.output_model, args.learning_rate)
