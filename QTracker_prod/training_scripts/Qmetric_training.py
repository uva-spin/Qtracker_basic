import ROOT
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def calculate_chi2(qpx, qpy, qpz, gpx, gpy, gpz):
    """
    Calculate the chi^2 value based on the scalar momentum difference.
    """
    qp = np.sqrt(qpx**2 + qpy**2 + qpz**2)  # Reconstructed momentum magnitude
    gp = np.sqrt(gpx**2 + gpy**2 + gpz**2)  # True momentum magnitude
    chi2 = (qp - gp)**2 / gp  # Chi^2 based on scalar momentum difference
    return chi2

def load_root_file(root_file):
    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")
    
    chi2_values, hit_arrays, momentum_vectors = [], [], []
    for event in tree:
        # Reconstructed momentum components
        qpx = event.qpx  # Array of size 2: [mu+, mu-]
        qpy = event.qpy  # Array of size 2: [mu+, mu-]
        qpz = event.qpz  # Array of size 2: [mu+, mu-]
        
        # True momentum components
        gpx = event.gpx  # Array of size 2: [mu+, mu-]
        gpy = event.gpy  # Array of size 2: [mu+, mu-]
        gpz = event.gpz  # Array of size 2: [mu+, mu-]
        
        # Hit arrays
        hit_array_mup = np.array(event.qHitArray_mup, dtype=np.float32)  # Hit array for mu+
        hit_array_mum = np.array(event.qHitArray_mum, dtype=np.float32)  # Hit array for mu-
        
        # Calculate chi^2 for mu+ and mu- tracks
        chi2_mup = calculate_chi2(qpx[0], qpy[0], qpz[0], gpx[0], gpy[0], gpz[0])
        chi2_mum = calculate_chi2(qpx[1], qpy[1], qpz[1], gpx[1], gpy[1], gpz[1])
        
        # Append chi^2 values, hit arrays, and momentum vectors
        chi2_values.extend([chi2_mup, chi2_mum])
        hit_arrays.extend([hit_array_mup, hit_array_mum])
        momentum_vectors.extend([[qpx[0], qpy[0], qpz[0]], [qpx[1], qpy[1], qpz[1]]])
    
    f.Close()
    
    return np.array(chi2_values), np.array(hit_arrays), np.array(momentum_vectors)

def train_chi2_model(chi2_values, hit_arrays, momentum_vectors):
    # Normalize input data
    scaler_hits = StandardScaler()
    hit_arrays_normalized = scaler_hits.fit_transform(hit_arrays)
    
    scaler_momentum = StandardScaler()
    momentum_vectors_normalized = scaler_momentum.fit_transform(momentum_vectors)
    
    # Combine hit arrays and momentum vectors into a single input array
    X = np.hstack((hit_arrays_normalized, momentum_vectors_normalized))
    
    # Split data into training and testing sets (shuffling is applied here)
    X_train, X_test, y_train, y_test = train_test_split(
        X, chi2_values, test_size=0.2, random_state=42, shuffle=True
    )
    print(chi2_values)
    # Debug: Check the distribution of mu+ and mu- tracks
    print(f"Training set size: {len(y_train)}")
    print(f"Testing set size: {len(y_test)}")
    
    # Build and train the model
    model = build_chi2_model(input_shape=(X_train.shape[1],))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=1000, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=64,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")
    
    return model

def build_chi2_model(input_shape):
    """Build a neural network model with regularization and dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def main(root_file):
    """
    Main function to calculate chi^2, train a model, and save it.
    """
    # Load data from ROOT file
    chi2_values, hit_arrays, momentum_vectors = load_root_file(root_file)
    
    # Train a model to predict chi^2
    model = train_chi2_model(chi2_values, hit_arrays, momentum_vectors)
    
    # Save the trained model
    model.save("chi2_predictor_model.h5")
    print("Chi^2 predictor model saved as 'chi2_predictor_model.h5'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate chi^2 and train a model to predict it.")
    parser.add_argument("root_file", type=str, help="Path to input ROOT file.")
    args = parser.parse_args()
    
    main(args.root_file)
