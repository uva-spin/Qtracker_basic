
import uproot
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Flatten
from tensorflow.keras.optimizers import AdamW
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
def build_model(input_shape, dropout_rate=0.0):
    # Input layer for the 3D array
    input_layer = Input(shape=input_shape)

    # Flatten the input to pass through dense layers
    x = Flatten()(input_layer)

    # Dense layers
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)

    # Output layer for gpx, gpy, gpz
    output = Dense(3, activation="linear")(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output)
    return model

# Train the model
def train_model(args):
    # Load data
    X, y = load_data(args.input_root)

    # Train val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Baseline MAE
    baseline_pred = np.mean(y_train, axis=0)
    baseline_mae = np.mean(np.abs(y_val - baseline_pred))
    print("Baseline MAE:", baseline_mae)

    # Build model
    model = build_model(input_shape=(62, 2), dropout_rate=args.dropout_rate)  # Input shape is (62, 2)
    model.summary()

    # Compile model
    model.compile(
        optimizer=AdamW(
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0
        ),
        loss="mse", 
        metrics=["mae"]
    )

    # Early stopping to prevent severe overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # LR scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=3, 
        min_lr=1e-6
    )

    # Train model
    model.fit(
        X_train, 
        y_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        validation_data=(X_val, y_val), 
        verbose=2, 
        callbacks=[early_stopping, lr_scheduler]
    )

    # Save model
    model.save(args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DNN to predict gpx, gpy, gpz from HitArray.")
    parser.add_argument("input_root", type=str, nargs='+', help="Path(s) to the input ROOT file(s).")
    parser.add_argument("--output", type=str, default="./models/mom_model.h5", help="Name of the output H5 model file.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for regularization.")
    args = parser.parse_args()
    
    train_model(args)

    