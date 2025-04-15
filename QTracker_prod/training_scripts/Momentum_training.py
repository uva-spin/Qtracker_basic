
import uproot
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Flatten
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy Adam optimizer for M1/M2 Mac compatibility
import argparse

# Function to load data from multiple ROOT files
def load_data(root_files):
    hit_arrays_list = []
    targets_list = []
    
    for root_file in root_files:
        with uproot.open(root_file) as file:
            tree = file["tree"]

            # Read in HitArray (with drift distances) and target variables
            hit_arrays = np.array(tree["HitArray"].array(library="np"), dtype=np.float32)
            gpx = tree["gpx"].array(library="ak").to_numpy().astype(np.float32)
            gpy = tree["gpy"].array(library="ak").to_numpy().astype(np.float32)
            gpz = tree["gpz"].array(library="ak").to_numpy().astype(np.float32)

            # Ensure they have the same number of samples
            assert hit_arrays.shape[0] == gpx.shape[0] == gpy.shape[0] == gpz.shape[0], \
                f"Shape mismatch in {root_file}: HitArray={hit_arrays.shape}, gpx={gpx.shape}, gpy={gpy.shape}, gpz={gpz.shape}"

            # Zero out irrelevant slots (both elementID and driftDistance)
            hit_arrays[:, 7:12] = 0     # unused station-1
            hit_arrays[:, 55:58] = 0    # DP-1
            hit_arrays[:, 59:62] = 0    # DP-2

            # Ensure driftDistance is zeroed out where elementID is zero
            hit_arrays[hit_arrays[:, :, 0] == 0, 1] = 0  # Zero out driftDistance where elementID is zero
            # Stack targets properly
            targets = np.column_stack((gpx, gpy, gpz))

            # Append to lists
            hit_arrays_list.append(hit_arrays)
            targets_list.append(targets)
    
    # Concatenate all data
    X = np.concatenate(hit_arrays_list, axis=0)
    y = np.concatenate(targets_list, axis=0)
    
    return X, y

# Build the model to accept a single 3D input
def build_model(input_shape):
    # Input layer for the 3D array
    input_layer = Input(shape=input_shape)

    # Flatten the input to pass through dense layers
    x = Flatten()(input_layer)

    # Dense layers
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)

    # Output layer for gpx, gpy, gpz
    output = Dense(3, activation="linear")(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# Train the model
def train_model(root_files, output_h5, epochs=100, batch_size=32):
    # Load data
    X, y = load_data(root_files)

    # Build model
    model = build_model(input_shape=(62, 2))  # Input shape is (62, 2)

    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # Save model
    model.save(output_h5)
    print(f"Model saved to {output_h5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DNN to predict gpx, gpy, gpz from HitArray.")
    parser.add_argument("input_root", type=str, nargs='+', help="Path(s) to the input ROOT file(s).")
    parser.add_argument("--output", type=str, default="./models/mom_model.h5", help="Name of the output H5 model file.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()
    
    train_model(args.input_root, args.output, epochs=args.epochs, batch_size=args.batch_size)

    
