import ROOT
import numpy as np
import tensorflow as tf
import argparse
import os

# Paths to models
MODEL_PATH_TRACK = "./models/track_finder.h5"
MODEL_PATH_MOMENTUM_MUP = "./models/mom_mup.h5"
MODEL_PATH_MOMENTUM_MUM = "./models/mom_mum.h5"
MODEL_PATH_METRIC = "./models/chi2_predictor_model.h5"

# Number of detectors and element IDs
NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

def custom_loss(y_true, y_pred):
    # Need to load custom loss so track finder model works
    """Custom loss function for model training."""
    y_muPlus_true, y_muMinus_true = tf.split(y_true, 2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, 2, axis=1)
    
    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)
    
    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)
    
    return tf.reduce_mean(loss_mup + loss_mum + 0.1 * overlap_penalty)


def load_detector_element_data(root_file):
    #Loads detectorID, elementID, and driftDistance for hit arrya refinement, adding drift distance and chi2 predictions.
    
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Error: Input ROOT file '{root_file}' not found.")

    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")
    if not tree:
        raise ValueError(f"Error: 'tree' not found in {root_file}.")

    detectorIDs, elementIDs, driftDistances = [], [], []
    for event in tree:
        detectorIDs.append(np.array(event.detectorID, dtype=np.int32))
        elementIDs.append(np.array(event.elementID, dtype=np.int32))
        driftDistances.append(np.array(event.driftDistance, dtype=np.float32))
    
    return detectorIDs, elementIDs, driftDistances, f


def load_data(root_file):
    #Loads detector hits from ROOT file to make hit matrix for track finding.
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Error: Input ROOT file '{root_file}' not found.")

    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")
    if not tree:
        raise ValueError(f"Error: 'tree' not found in {root_file}.")

    X, event_entries = [], []
    for i, event in enumerate(tree):
        event_hits_matrix = np.zeros((NUM_DETECTORS, NUM_ELEMENT_IDS), dtype=np.float32)
        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < NUM_DETECTORS and 0 <= elem_id < NUM_ELEMENT_IDS:
                event_hits_matrix[det_id, elem_id] = 1
        X.append(event_hits_matrix)
        event_entries.append(i)
    
    return np.expand_dims(np.array(X), axis=-1), event_entries, f


def predict_hit_arrays(model, X):
    """Runs the TrackFinder model to predict hit arrays."""
    predictions = model.predict(X)
    return np.argmax(predictions[:, 0, :, :], axis=-1), np.argmax(predictions[:, 1, :, :], axis=-1)


def predict_momentum(hit_arrays, model):
    """
    Predicts px, py, pz using a trained model after masking unused slots.
    Validates that the input is a 3D array with shape (num_events, 62, 2).
    """
    # Validate input shape
    if len(hit_arrays.shape) != 3 or hit_arrays.shape[1] != 62 or hit_arrays.shape[2] != 2:
        raise ValueError(
            f"Input shape {hit_arrays.shape} is invalid. "
            f"Expected shape: (num_events, 62, 2). "
            f"The input must include both elementID and driftDistance."
        )
    
    # Mask unused slots (if necessary)
    hit_arrays[:, 7:12, :] = 0     # unused station-1
    hit_arrays[:, 55:58, :] = 0    # DP-1
    hit_arrays[:, 59:62, :] = 0    # DP-2
    
    return model.predict(hit_arrays)



def predict_chi2(hit_arrays, momentum_vectors, chi2_model_path=MODEL_PATH_METRIC):
    """
    Predicts chi^2 values for the given hit arrays and momentum vectors using a pre-trained model.

    Args:
        hit_arrays (np.ndarray): Hit arrays for tracks (shape: [num_tracks, num_detectors]).
        momentum_vectors (np.ndarray): Momentum vectors (shape: [num_tracks, 3]).
        chi2_model_path (str): Path to the trained chi^2 prediction model.

    Returns:
        np.ndarray: Predicted chi^2 values for the tracks.
    """
    # Load the trained chi^2 prediction model
    chi2_model = tf.keras.models.load_model(chi2_model_path)
    
    # Combine hit arrays and momentum vectors into a single input array
    X = np.hstack((hit_arrays, momentum_vectors))

    # Predict chi^2 values
    chi2_predictions = chi2_model.predict(X)

    return chi2_predictions.flatten()  # Flatten to 1D array


def write_predicted_root_file(output_file, input_file, rHitArray_mup, rHitArray_mum, results, event_entries, chi2_mup, chi2_mum):
    """Writes predictions to a new ROOT file, preserving the original data and storing hit arrays."""
    f_input = ROOT.TFile.Open(input_file, "READ")
    tree_input = f_input.Get("tree")
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)
    output_tree = tree_input.CloneTree(0)

    muID = ROOT.std.vector("int")()
    HitArray_mup = np.zeros(62, dtype=np.int32)
    HitArray_mum = np.zeros(62, dtype=np.int32)
    DriftDistance_mup = np.zeros(62, dtype=np.float32)
    DriftDistance_mum = np.zeros(62, dtype=np.float32)
    qpx = np.zeros(2, dtype=np.float32)
    qpy = np.zeros(2, dtype=np.float32)
    qpz = np.zeros(2, dtype=np.float32)
    qchi2 = np.zeros(2, dtype=np.float32)  # New branch for chi^2 values

    output_tree.Branch("muID", muID)
    output_tree.Branch("qHitArray_mup", HitArray_mup, "qHitArray_mup[62]/I")
    output_tree.Branch("qHitArray_mum", HitArray_mum, "qHitArray_mum[62]/I")
    output_tree.Branch("driftDistance_mup", DriftDistance_mup, "driftDistance_mup[62]/F")
    output_tree.Branch("driftDistance_mum", DriftDistance_mum, "driftDistance_mum[62]/F")
    output_tree.Branch("qpx", qpx, "qpx[2]/F")
    output_tree.Branch("qpy", qpy, "qpy[2]/F")
    output_tree.Branch("qpz", qpz, "qpz[2]/F")
    output_tree.Branch("qchi2", qchi2, "qchi2[2]/F")  # New branch for chi^2 values

    for i, entry_idx in enumerate(event_entries):
        tree_input.GetEntry(entry_idx)
        
        # Fill hit arrays and drift distances
        HitArray_mup[:] = rHitArray_mup[i, :, 0]  # ElementIDs
        HitArray_mum[:] = rHitArray_mum[i, :, 0]  # ElementIDs
        DriftDistance_mup[:] = rHitArray_mup[i, :, 1]  # Drift distances
        DriftDistance_mum[:] = rHitArray_mum[i, :, 1]  # Drift distances
        
        muID.clear()
        muID.push_back(1)
        muID.push_back(2)
        qpx[0], qpy[0], qpz[0] = results['momentum_mup'][i]
        qpx[1], qpy[1], qpz[1] = results['momentum_mum'][i]
        qchi2[0] = chi2_mup[i]  
        qchi2[1] = chi2_mum[i]  

        output_tree.Fill()

    fout.Write()
    fout.Close()
    f_input.Close()
    print(f"Predicted data written to {output_file}, retaining all original data.")

def refine_hit_arrays(hit_array_mup, hit_array_mum, detectorIDs, elementIDs):
    """
    Refines the HitArrays by replacing inferred elementIDs with the closest actual elementID
    using the detectorID and elementID vectors. Returns 0 if no actual hits exist.
    Optimized for speed.
    """
    def find_closest_actual_hit(detector_id, inferred_element, detectorIDs_event, elementIDs_event):
        """
        Finds the closest actual hit to the inferred_element for a specific detector_id.
        Returns 0 if no hits exist.
        """
        if inferred_element == 0:
            return 0  # Preserve 0 values (no hit).

        # Filter elementIDs for the given detector_id
        actual_elementIDs = elementIDs_event[detectorIDs_event == detector_id]

        if len(actual_elementIDs) == 0:
            return 0  # Return 0 if no hits exist.

        # Find the closest actual hit elementID using NumPy's vectorized operations
        closest_elementID = actual_elementIDs[np.argmin(np.abs(actual_elementIDs - inferred_element))]

        return closest_elementID

    # Initialize refined arrays
    refined_mup = np.zeros_like(hit_array_mup)
    refined_mum = np.zeros_like(hit_array_mum)

    num_events, num_detectors = hit_array_mup.shape

    # Precompute detector IDs (1-based to match detector_id in the ROOT file)
    detector_ids = np.arange(1, num_detectors + 1)

    # Iterate over events
    for event in range(num_events):
        # Convert detectorIDs and elementIDs to NumPy arrays for faster processing
        detectorIDs_event = np.array(detectorIDs[event], dtype=np.int32)
        elementIDs_event = np.array(elementIDs[event], dtype=np.int32)

        # Iterate over detectors
        for detector in range(num_detectors):
            # Get inferred elementIDs for mu+ and mu-
            inferred_mup = hit_array_mup[event, detector]
            inferred_mum = hit_array_mum[event, detector]

            # Find the closest actual hits
            refined_mup[event, detector] = find_closest_actual_hit(
                detector_ids[detector], inferred_mup, detectorIDs_event, elementIDs_event
            )
            refined_mum[event, detector] = find_closest_actual_hit(
                detector_ids[detector], inferred_mum, detectorIDs_event, elementIDs_event
            )

    return refined_mup, refined_mum

def write_hit_arrays_to_file(rHitArray_mup, rHitArray_mum, refined_HitArray_mup, refined_HitArray_mum, output_file):
    """
    Writes the original and refined hit arrays to a text file for debugging.
    """
    with open(output_file, "w") as f:
        num_events = rHitArray_mup.shape[0]
        for event in range(num_events):
            f.write(f"Event {event}:\n")
            
            # Write mu+ (mup) arrays
            f.write("  mu+ (mup):\n")
            f.write("    Slot | Original rHitArray_mup | Refined HitArray_mup\n")
            f.write("    -----------------------------------------------\n")
            for slot in range(NUM_DETECTORS):
                original_value = rHitArray_mup[event, slot]
                refined_value = refined_HitArray_mup[event, slot]
                f.write(f"    {slot:4} | {original_value:21} | {refined_value:21}\n")
            
            # Write mu- (mum) arrays
            f.write("\n  mu- (mum):\n")
            f.write("    Slot | Original rHitArray_mum | Refined HitArray_mum\n")
            f.write("    -----------------------------------------------\n")
            for slot in range(NUM_DETECTORS):
                original_value = rHitArray_mum[event, slot]
                refined_value = refined_HitArray_mum[event, slot]
                f.write(f"    {slot:4} | {original_value:21} | {refined_value:21}\n")
            
            f.write("\n")  # Add a blank line between events


def add_drift_distance_to_hit_arrays(refined_mup, refined_mum, detectorIDs, elementIDs, driftDistances):
    """
    Adds drift distances to the refined hit arrays by looking up the driftDistance for each real elementID.
    Expands the hit arrays by 1 dimension to hold the driftDistance.

    Args:
        refined_mup (np.ndarray): Refined hit array for mu+ (shape: [num_events, num_detectors]).
        refined_mum (np.ndarray): Refined hit array for mu- (shape: [num_events, num_detectors]).
        detectorIDs (list of np.ndarray): List of detectorID vectors for each event.
        elementIDs (list of np.ndarray): List of elementID vectors for each event.
        driftDistances (list of np.ndarray): List of driftDistance vectors for each event.

    Returns:
        np.ndarray: Refined hit array for mu+ with drift distances (shape: [num_events, num_detectors, 2]).
        np.ndarray: Refined hit array for mu- with drift distances (shape: [num_events, num_detectors, 2]).
    """
    num_events, num_detectors = refined_mup.shape

    # Initialize expanded hit arrays with an additional dimension for drift distance
    refined_mup_with_drift = np.zeros((num_events, num_detectors, 2), dtype=np.float32)
    refined_mum_with_drift = np.zeros((num_events, num_detectors, 2), dtype=np.float32)

    # Iterate over events
    for event in range(num_events):
        # Convert detectorIDs, elementIDs, and driftDistances to NumPy arrays for faster processing
        detectorIDs_event = np.array(detectorIDs[event], dtype=np.int32)
        elementIDs_event = np.array(elementIDs[event], dtype=np.int32)
        driftDistances_event = np.array(driftDistances[event], dtype=np.float32)

        # Iterate over detectors
        for detector in range(num_detectors):
            # Get the real elementID for mu+ and mu-
            element_mup = refined_mup[event, detector]
            element_mum = refined_mum[event, detector]

            # Look up drift distance for mu+
            if element_mup == 0:
                drift_mup = 0.0  # No hit, drift distance is 0
            else:
                # Find the index of the matching detectorID and elementID
                match_index = np.where((detectorIDs_event == detector + 1) & (elementIDs_event == element_mup))[0]
                if len(match_index) > 0:
                    drift_mup = driftDistances_event[match_index[0]]
                else:
                    drift_mup = 0.0  # No matching hit, drift distance is 0

            # Look up drift distance for mu-
            if element_mum == 0:
                drift_mum = 0.0  # No hit, drift distance is 0
            else:
                # Find the index of the matching detectorID and elementID
                match_index = np.where((detectorIDs_event == detector + 1) & (elementIDs_event == element_mum))[0]
                if len(match_index) > 0:
                    drift_mum = driftDistances_event[match_index[0]]
                else:
                    drift_mum = 0.0  # No matching hit, drift distance is 0

            # Store the elementID and drift distance in the expanded hit arrays
            refined_mup_with_drift[event, detector, 0] = element_mup
            refined_mup_with_drift[event, detector, 1] = drift_mup
            refined_mum_with_drift[event, detector, 0] = element_mum
            refined_mum_with_drift[event, detector, 1] = drift_mum

    return refined_mup_with_drift, refined_mum_with_drift


def process_data(root_file, output_file="tracker_output.root", use_chi2_model=False):
    """Loads models, predicts hit arrays and momentum, refines hit arrays, and writes to a new ROOT file."""
    with tf.keras.utils.custom_object_scope({"custom_loss": custom_loss, "Adam": tf.keras.optimizers.legacy.Adam}):
        model_track = tf.keras.models.load_model(MODEL_PATH_TRACK)

    model_momentum_mup = tf.keras.models.load_model(MODEL_PATH_MOMENTUM_MUP)
    model_momentum_mum = tf.keras.models.load_model(MODEL_PATH_MOMENTUM_MUM)

    X, event_entries, _ = load_data(root_file)
    detectorIDs, elementIDs, driftDistances, _ = load_detector_element_data(root_file)

    rHitArray_mup, rHitArray_mum = predict_hit_arrays(model_track, X)

    refined_HitArray_mup, refined_HitArray_mum = refine_hit_arrays(
        rHitArray_mup, rHitArray_mum, detectorIDs, elementIDs
    )

    refined_HitArray_mup_with_drift, refined_HitArray_mum_with_drift = add_drift_distance_to_hit_arrays(
        refined_HitArray_mup, refined_HitArray_mum, detectorIDs, elementIDs, driftDistances
    )

    results = {
        'momentum_mup': predict_momentum(refined_HitArray_mup_with_drift, model_momentum_mup),
        'momentum_mum': predict_momentum(refined_HitArray_mum_with_drift, model_momentum_mum)
    }

    # chi^2 computation (optional)
    if use_chi2_model:
        combined_hit_arrays = np.concatenate([refined_HitArray_mup, refined_HitArray_mum], axis=0)
        combined_momentum_vectors = np.concatenate([results['momentum_mup'], results['momentum_mum']], axis=0)
        chi2_predictions = predict_chi2(combined_hit_arrays, combined_momentum_vectors, chi2_model_path=MODEL_PATH_METRIC)
        num_events = len(refined_HitArray_mup)
        chi2_mup = chi2_predictions[:num_events]
        chi2_mum = chi2_predictions[num_events:]
    else:
        chi2_mup = np.zeros(len(refined_HitArray_mup))
        chi2_mum = np.zeros(len(refined_HitArray_mum))

    write_predicted_root_file(
        output_file,
        root_file,
        refined_HitArray_mup_with_drift,
        refined_HitArray_mum_with_drift,
        results,
        event_entries,
        chi2_mup,
        chi2_mum
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run track finder and momentum inference on ROOT file.")
    parser.add_argument("root_file", type=str, help="Path to input ROOT file.")
    parser.add_argument("--output_file", type=str, default="qtracker_reco.root", help="Output ROOT file.")
    args = parser.parse_args()
    
    process_data(args.root_file, args.output_file)

