# TODO: Modify to support new multi-track evaluation

import ROOT
import numpy as np
import tensorflow as tf
import argparse
import os
from ROOT import TTree, TMatrixD
from numba import njit, prange
from tensorflow.keras.losses import MeanSquaredError


from refine import refine_hit_arrays
from models.layers import AxialAttention

# USE_CHI2 must be False the first time the script is ran to obtain output for training the quality metric
USE_CHI2 = False
USE_DECLUSTERING = False  # Toggle to enable/disable declustering
USE_SMAXMATRIX = False  # Toggle to enable/disable write softmax matrix

# Paths to model checkpoints
MODEL_PATH_TRACK = "./checkpoints/track_finder_flagship.keras"
MODEL_PATH_MOMENTUM_MUP = "./checkpoints/mom_mup.h5"
MODEL_PATH_MOMENTUM_MUM = "./checkpoints/mom_mum.h5"
MODEL_PATH_METRIC = "./checkpoints/chi2_predictor.h5"

# Number of detectors and element IDs
NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def load_detector_element_data(root_file):
    """
    Loads detectorID, elementID, driftDistance, and tdcTime from ROOT file.
    """
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Error: Input ROOT file '{root_file}' not found.")

    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")
    if not tree:
        raise ValueError(f"Error: 'tree' not found in {root_file}.")

    detectorIDs, elementIDs, driftDistances, tdcTimes = [], [], [], []

    for event in tree:
        detectorIDs.append(np.array(event.detectorID, dtype=np.int32))
        elementIDs.append(np.array(event.elementID, dtype=np.int32))
        driftDistances.append(np.array(event.driftDistance, dtype=np.float32))
        tdcTimes.append(
            np.array(event.tdcTime, dtype=np.float32)
        )  # <-- this was missing

    return detectorIDs, elementIDs, driftDistances, tdcTimes, f


@njit(parallel=True)
def declusterize(hits, drift, tdc):
    """
    Removes clustered or noisy hits from the hit matrix.
    Operates in-place on hits, drift, and tdc arrays.

    Args:
        hits: 3D array of shape (num_events, 62, 201), binary hit matrix.
        drift: 3D array, same shape, with drift distances.
        tdc: 3D array, same shape, with TDC values.
    """
    num_events, num_detectors, num_elements = hits.shape

    for k in prange(num_events):
        for i in range(31):  # You may increase this to 62 if desired
            for j in range(0, 199):  # avoid out-of-bounds at j+2
                if hits[k, i, j] == 1 and hits[k, i, j + 1] == 1:
                    if hits[k, i, j + 2] == 0:  # 2-hit cluster
                        if drift[k, i, j] > 0.9 and drift[k, i, j + 1] > 0.4:
                            hits[k, i, j] = 0
                            drift[k, i, j] = 0
                            tdc[k, i, j] = 0
                        elif drift[k, i, j + 1] > 0.9 and drift[k, i, j] > 0.4:
                            hits[k, i, j + 1] = 0
                            drift[k, i, j + 1] = 0
                            tdc[k, i, j + 1] = 0
                        if abs(tdc[k, i, j] - tdc[k, i, j + 1]) < 8:
                            hits[k, i, j] = 0
                            hits[k, i, j + 1] = 0
                            drift[k, i, j] = 0
                            drift[k, i, j + 1] = 0
                            tdc[k, i, j] = 0
                            tdc[k, i, j + 1] = 0
                    else:
                        n = 2
                        while j + n < num_elements and hits[k, i, j + n] == 1:
                            n += 1
                        dt_mean = 0.0
                        for m in range(n - 1):
                            dt_mean += abs(tdc[k, i, j + m] - tdc[k, i, j + m + 1])
                        dt_mean /= n - 1
                        if dt_mean < 10:
                            for m in range(n):
                                hits[k, i, j + m] = 0
                                drift[k, i, j + m] = 0
                                tdc[k, i, j + m] = 0

            # Now do a right-to-left pass
            for j in range(0, 199):
                jj = 200 - j
                if hits[k, i, jj] == 1 and hits[k, i, jj - 1] == 1:
                    if hits[k, i, jj - 2] == 0:
                        if drift[k, i, jj] > 0.9 and drift[k, i, jj - 1] > 0.4:
                            hits[k, i, jj] = 0
                            drift[k, i, jj] = 0
                        elif drift[k, i, jj - 1] > 0.9 and drift[k, i, jj] > 0.4:
                            hits[k, i, jj - 1] = 0
                            drift[k, i, jj - 1] = 0
                        if abs(tdc[k, i, jj] - tdc[k, i, jj - 1]) < 8:
                            hits[k, i, jj] = 0
                            hits[k, i, jj - 1] = 0
                            drift[k, i, jj] = 0
                            drift[k, i, jj - 1] = 0
                            tdc[k, i, jj] = 0
                            tdc[k, i, jj - 1] = 0
                    else:
                        n = 2
                        while jj - n >= 0 and hits[k, i, jj - n] == 1:
                            n += 1
                        dt_mean = 0.0
                        for m in range(n - 1):
                            dt_mean += abs(tdc[k, i, jj - m] - tdc[k, i, jj - m - 1])
                        dt_mean /= n - 1
                        if dt_mean < 10:
                            for m in range(n):
                                hits[k, i, jj - m] = 0
                                drift[k, i, jj - m] = 0
                                tdc[k, i, jj - m] = 0


def build_hit_drift_tdc_matrices(detectorIDs, elementIDs, driftDistances, tdcTimes):
    """
    Constructs 3D hit, drift, and tdc matrices (num_events, 62, 201).
    Each hit location will be 1 in hit matrix, and corresponding values in drift and tdc.
    """
    num_events = len(detectorIDs)
    hits = np.zeros((num_events, NUM_DETECTORS, NUM_ELEMENT_IDS), dtype=np.int32)
    drift = np.zeros_like(hits, dtype=np.float32)
    tdc = np.zeros_like(hits, dtype=np.float32)

    for i in range(num_events):
        for det, elem, d, t in zip(
            detectorIDs[i], elementIDs[i], driftDistances[i], tdcTimes[i]
        ):
            if 0 <= det < NUM_DETECTORS and 0 <= elem < NUM_ELEMENT_IDS:
                hits[i, det, elem] = 1
                drift[i, det, elem] = d
                tdc[i, det, elem] = t

    return hits, drift, tdc


def predict_hit_arrays(model, X, pair_idx=0):
    """
    Runs MultiTrackFinder and extracts predictions for a specific pair.

    Args:
        model: Trained MultiTrackFinder model
        X: Input hit matrix (num_events, 62, 201, 1)
        pair_idx: Which pair to extract (0 to max_pairs-1)

    Returns:
        rHitArray_mup: (num_events, 62) - argmax elementIDs for μ⁺
        rHitArray_mum: (num_events, 62) - argmax elementIDs for μ⁻
        softmax_mup: (num_events, 62, 201) - softmax probabilities for μ⁺
        softmax_mum: (num_events, 62, 201) - softmax probabilities for μ⁻
    """
    print(f"Running hit array predictions for pair {pair_idx}...")
    preds = []
    chunk_size = 128

    for i in range(0, len(X), chunk_size):
        X_chunk = tf.cast(X[i : i + chunk_size], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        # y_chunk[1] shape: (chunk, max_pairs, 2, 62, 201)
        preds.append(y_chunk[1])

    predictions = np.concatenate(preds, axis=0)
    # Shape: (num_events, max_pairs, 2, 62, 201)

    # Extract specific pair
    softmax_mup = predictions[:, pair_idx, 0, :, :]  # (num_events, 62, 201)
    softmax_mum = predictions[:, pair_idx, 1, :, :]  # (num_events, 62, 201)

    rHitArray_mup = np.argmax(softmax_mup, axis=-1)  # (num_events, 62)
    rHitArray_mum = np.argmax(softmax_mum, axis=-1)  # (num_events, 62)

    return rHitArray_mup, rHitArray_mum, softmax_mup, softmax_mum


def predict_momentum(hit_arrays, model):
    """
    Predicts px, py, pz using a trained model after masking unused slots.
    Validates that the input is a 3D array with shape (num_events, 62, 2).
    """
    # Validate input shape
    if (
        len(hit_arrays.shape) != 3
        or hit_arrays.shape[1] != 62
        or hit_arrays.shape[2] != 2
    ):
        raise ValueError(
            f"Input shape {hit_arrays.shape} is invalid. "
            f"Expected shape: (num_events, 62, 2). "
            f"The input must include both elementID and driftDistance."
        )

    # Mask unused slots (if necessary)
    hit_arrays[:, 7:12, :] = 0  # unused station-1
    hit_arrays[:, 55:58, :] = 0  # DP-1
    hit_arrays[:, 59:62, :] = 0  # DP-2

    print("Running momentum predictions...")
    return model.predict(hit_arrays, verbose=0)


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
    print("Running chi^2 predictions...")
    chi2_predictions = chi2_model.predict(X, verbose=0)

    return chi2_predictions.flatten()  # Flatten to 1D array


def write_predicted_root_file(
    output_file,
    input_file,
    rHitArray_mup,
    rHitArray_mum,
    results,
    event_entries,
    chi2_mup,
    chi2_mum,
    hits_before=None,
    hits_after=None,
    softmax_mup=None,
    softmax_mum=None,
):
    """Writes predictions to a new ROOT file, preserving the original data and storing hit arrays."""
    f_input = ROOT.TFile.Open(input_file, "READ")
    tree_input = f_input.Get("tree")
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    if hits_before is not None and hits_after is not None:
        write_hit_matrices_to_root(fout, hits_before, hits_after)

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
    output_tree.Branch(
        "driftDistance_mup", DriftDistance_mup, "driftDistance_mup[62]/F"
    )
    output_tree.Branch(
        "driftDistance_mum", DriftDistance_mum, "driftDistance_mum[62]/F"
    )
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
        qpx[0], qpy[0], qpz[0] = results["momentum_mup"][i]
        qpx[1], qpy[1], qpz[1] = results["momentum_mum"][i]
        qchi2[0] = chi2_mup[i]
        qchi2[1] = chi2_mum[i]

        output_tree.Fill()

    if softmax_mup is not None and softmax_mum is not None:
        write_softmax_to_root(fout, softmax_mup, softmax_mum)

    fout.Write()
    fout.Close()
    f_input.Close()
    print(f"Predicted data written to {output_file}, retaining all original data.")


def write_hit_matrices_to_root(fout, hits_before, hits_after):
    """
    Writes the hit matrices before and after declustering into an already open ROOT file.

    Args:
        fout: Open ROOT TFile where the tree should be written.
        hits_before: 3D numpy array (num_events, 62, 201) before cleaning
        hits_after: 3D numpy array (num_events, 62, 201) after cleaning
    """
    tree = TTree("hitMatrixTree", "Hit matrices before and after declustering")

    mat_before = TMatrixD(NUM_DETECTORS, NUM_ELEMENT_IDS)
    mat_after = TMatrixD(NUM_DETECTORS, NUM_ELEMENT_IDS)

    tree.Branch("hitMatrix_before", mat_before)
    tree.Branch("hitMatrix_after", mat_after)

    num_events = hits_before.shape[0]
    for i in range(num_events):
        for det in range(NUM_DETECTORS):
            for elem in range(NUM_ELEMENT_IDS):
                mat_before[det][elem] = hits_before[i, det, elem]
                mat_after[det][elem] = hits_after[i, det, elem]
        tree.Fill()

    tree.Write()  # <<== VERY IMPORTANT! Write the tree into fout
    print(f"Hit matrices written to {fout.GetName()}")


def write_softmax_to_root(fout, softmax_mup, softmax_mum):
    """
    Writes the softmax response matrices for mu+ and mu- into the ROOT file.
    """
    tree = TTree("softmaxTree", "Softmax output matrices for mup and mum")

    mat_softmax_mup = TMatrixD(NUM_DETECTORS, NUM_ELEMENT_IDS)
    mat_softmax_mum = TMatrixD(NUM_DETECTORS, NUM_ELEMENT_IDS)

    tree.Branch("softmax_mup", mat_softmax_mup)
    tree.Branch("softmax_mum", mat_softmax_mum)

    num_events = softmax_mup.shape[0]
    for i in range(num_events):
        for det in range(NUM_DETECTORS):
            for elem in range(NUM_ELEMENT_IDS):
                mat_softmax_mup[det][elem] = softmax_mup[i, det, elem]
                mat_softmax_mum[det][elem] = softmax_mum[i, det, elem]
        tree.Fill()

    tree.Write()
    print(f"Softmax matrices written to {fout.GetName()}")


def write_hit_arrays_to_file(
    rHitArray_mup,
    rHitArray_mum,
    refined_HitArray_mup,
    refined_HitArray_mum,
    output_file,
):
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


def add_drift_distance_to_hit_arrays(
    refined_mup, refined_mum, detectorIDs, elementIDs, driftDistances
):
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

        assert (
            len(detectorIDs_event) == len(elementIDs_event) == len(driftDistances_event)
        ), (
            f"[Mismatch] event {event}: detectorIDs={len(detectorIDs_event)} elementIDs={len(elementIDs_event)} driftDistances={len(driftDistances_event)}"
        )

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
                match_index = np.where(
                    (detectorIDs_event == detector + 1)
                    & (elementIDs_event == element_mup)
                )[0]
                if len(match_index) > 0:
                    drift_mup = driftDistances_event[match_index[0]]
                else:
                    drift_mup = 0.0  # No matching hit, drift distance is 0

            # Look up drift distance for mu-
            if element_mum == 0:
                drift_mum = 0.0  # No hit, drift distance is 0
            else:
                # Find the index of the matching detectorID and elementID
                match_index = np.where(
                    (detectorIDs_event == detector + 1)
                    & (elementIDs_event == element_mum)
                )[0]
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


def process_data(
    root_file,
    output_file="qtracker_multi_reco.root",
    max_pairs=5,
    use_chi2_model=USE_CHI2,
):
    """
    Loads models, predicts hit arrays and momentum for multiple pairs,
    refines hit arrays, and writes to a new ROOT file.

    Args:
        root_file: Path to input ROOT file
        output_file: Path to output ROOT file
        max_pairs: Maximum number of dimuon pairs to reconstruct
        use_chi2_model: Whether to use chi2 predictor
    """
    custom_objects = {"AxialAttention": AxialAttention}
    model_track = tf.keras.models.load_model(
        MODEL_PATH_TRACK, compile=False, custom_objects=custom_objects
    )

    model_momentum_mup = tf.keras.models.load_model(
        MODEL_PATH_MOMENTUM_MUP, custom_objects={"mse": MeanSquaredError()}
    )
    model_momentum_mum = tf.keras.models.load_model(
        MODEL_PATH_MOMENTUM_MUM, custom_objects={"mse": MeanSquaredError()}
    )

    detectorIDs, elementIDs, driftDistances, tdcTimes, _ = load_detector_element_data(
        root_file
    )

    # Build raw matrices
    hits, drift, tdc = build_hit_drift_tdc_matrices(
        detectorIDs, elementIDs, driftDistances, tdcTimes
    )

    # Make a copy for comparison
    hits_before = hits.copy()  # noqa: F841

    # Decluster in place
    if USE_DECLUSTERING:
        print("Declustering enabled — cleaning hit matrix...")
        declusterize(hits, drift, tdc)
    else:
        print("Declustering disabled — using raw hit matrix.")

    # Use declustered hit matrix as CNN input
    X = np.expand_dims(hits, axis=-1)
    num_events = len(X)

    # Initialize storage for all pairs
    all_refined_mup = []  # List of (num_events, 62, 2) arrays
    all_refined_mum = []
    all_momentum_mup = []
    all_momentum_mum = []
    all_chi2_mup = []
    all_chi2_mum = []

    # Process each pair separately
    for pair_idx in range(max_pairs):
        print(f"\nProcessing pair {pair_idx}...")

        # Predict for this specific pair
        rHitArray_mup, rHitArray_mum, _, _ = predict_hit_arrays(
            model_track, X, pair_idx
        )

        # Refine (reuse existing single-pair function)
        refined_mup, refined_mum = refine_hit_arrays(
            rHitArray_mup, rHitArray_mum, detectorIDs, elementIDs
        )

        # Add drift distances
        refined_mup_drift, refined_mum_drift = add_drift_distance_to_hit_arrays(
            refined_mup, refined_mum, detectorIDs, elementIDs, driftDistances
        )

        # Predict momentum (reuse existing models)
        mom_mup = predict_momentum(refined_mup_drift, model_momentum_mup)
        mom_mum = predict_momentum(refined_mum_drift, model_momentum_mum)

        # Predict chi2 if enabled
        if use_chi2_model:
            combined_hit_arrays = np.concatenate([refined_mup, refined_mum], axis=0)
            combined_momentum_vectors = np.concatenate([mom_mup, mom_mum], axis=0)
            chi2_preds = predict_chi2(
                combined_hit_arrays,
                combined_momentum_vectors,
                chi2_model_path=MODEL_PATH_METRIC,
            )
            chi2_mup = chi2_preds[:num_events]
            chi2_mum = chi2_preds[num_events:]
        else:
            chi2_mup = np.zeros(num_events)
            chi2_mum = np.zeros(num_events)

        all_refined_mup.append(refined_mup_drift)
        all_refined_mum.append(refined_mum_drift)
        all_momentum_mup.append(mom_mup)
        all_momentum_mum.append(mom_mum)
        all_chi2_mup.append(chi2_mup)
        all_chi2_mum.append(chi2_mum)

    # Stack results: shape (num_events, max_pairs, 62, 2)
    refined_mup_all = np.stack(all_refined_mup, axis=1)
    refined_mum_all = np.stack(all_refined_mum, axis=1)
    momentum_mup_all = np.stack(all_momentum_mup, axis=1)  # (num_events, max_pairs, 3)
    momentum_mum_all = np.stack(all_momentum_mum, axis=1)
    chi2_mup_all = np.stack(all_chi2_mup, axis=1)  # (num_events, max_pairs)
    chi2_mum_all = np.stack(all_chi2_mum, axis=1)

    # Write to ROOT file with multi-pair format
    write_predicted_root_file_multi(
        output_file,
        root_file,
        refined_mup_all,
        refined_mum_all,
        momentum_mup_all,
        momentum_mum_all,
        chi2_mup_all,
        chi2_mum_all,
        max_pairs,
    )


def write_predicted_root_file_multi(
    output_file,
    input_file,
    rHitArray_mup_all,
    rHitArray_mum_all,
    momentum_mup_all,
    momentum_mum_all,
    chi2_mup_all,
    chi2_mum_all,
    max_pairs,
):
    """
    Writes multi-pair predictions to a new ROOT file, preserving the original data.

    Args:
        output_file: Path to output ROOT file
        input_file: Path to input ROOT file
        rHitArray_mup_all: (num_events, max_pairs, 62, 2) - μ⁺ elementIDs and drift
        rHitArray_mum_all: (num_events, max_pairs, 62, 2) - μ⁻ elementIDs and drift
        momentum_mup_all: (num_events, max_pairs, 3) - μ⁺ px, py, pz
        momentum_mum_all: (num_events, max_pairs, 3) - μ⁻ px, py, pz
        chi2_mup_all: (num_events, max_pairs) - μ⁺ chi2 values
        chi2_mum_all: (num_events, max_pairs) - μ⁻ chi2 values
        max_pairs: Maximum number of pairs
    """
    f_input = ROOT.TFile.Open(input_file, "READ")
    tree_input = f_input.Get("tree")
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)
    output_tree = tree_input.CloneTree(0)

    num_events = rHitArray_mup_all.shape[0]

    # Create 2D arrays for hit arrays and drift distances
    qHitArray_mup = np.zeros((max_pairs, NUM_DETECTORS), dtype=np.int32)
    qHitArray_mum = np.zeros((max_pairs, NUM_DETECTORS), dtype=np.int32)
    driftDistance_mup = np.zeros((max_pairs, NUM_DETECTORS), dtype=np.float32)
    driftDistance_mum = np.zeros((max_pairs, NUM_DETECTORS), dtype=np.float32)

    # Create vectors for momentum and chi2
    qpx = np.zeros(max_pairs * 2, dtype=np.float32)
    qpy = np.zeros(max_pairs * 2, dtype=np.float32)
    qpz = np.zeros(max_pairs * 2, dtype=np.float32)
    qchi2 = np.zeros(max_pairs * 2, dtype=np.float32)

    nPairs = np.zeros(1, dtype=np.int32)
    muID = ROOT.std.vector("int")()

    # Create branches
    output_tree.Branch("nPairs", nPairs, "nPairs/I")
    output_tree.Branch("muID", muID)
    output_tree.Branch(
        "qHitArray_mup", qHitArray_mup, f"qHitArray_mup[{max_pairs}][{NUM_DETECTORS}]/I"
    )
    output_tree.Branch(
        "qHitArray_mum", qHitArray_mum, f"qHitArray_mum[{max_pairs}][{NUM_DETECTORS}]/I"
    )
    output_tree.Branch(
        "driftDistance_mup",
        driftDistance_mup,
        f"driftDistance_mup[{max_pairs}][{NUM_DETECTORS}]/F",
    )
    output_tree.Branch(
        "driftDistance_mum",
        driftDistance_mum,
        f"driftDistance_mum[{max_pairs}][{NUM_DETECTORS}]/F",
    )
    output_tree.Branch("qpx", qpx, f"qpx[{max_pairs * 2}]/F")
    output_tree.Branch("qpy", qpy, f"qpy[{max_pairs * 2}]/F")
    output_tree.Branch("qpz", qpz, f"qpz[{max_pairs * 2}]/F")
    output_tree.Branch("qchi2", qchi2, f"qchi2[{max_pairs * 2}]/F")

    for i in range(num_events):
        tree_input.GetEntry(i)

        # Determine number of valid pairs (non-zero predictions)
        valid_pairs = 0
        for pair_idx in range(max_pairs):
            if np.any(rHitArray_mup_all[i, pair_idx, :, 0] != 0) or np.any(
                rHitArray_mum_all[i, pair_idx, :, 0] != 0
            ):
                valid_pairs = pair_idx + 1

        nPairs[0] = valid_pairs

        # Fill hit arrays and drift distances for all pairs
        for pair_idx in range(max_pairs):
            qHitArray_mup[pair_idx, :] = rHitArray_mup_all[
                i, pair_idx, :, 0
            ]  # ElementIDs
            qHitArray_mum[pair_idx, :] = rHitArray_mum_all[
                i, pair_idx, :, 0
            ]  # ElementIDs
            driftDistance_mup[pair_idx, :] = rHitArray_mup_all[
                i, pair_idx, :, 1
            ]  # Drift
            driftDistance_mum[pair_idx, :] = rHitArray_mum_all[
                i, pair_idx, :, 1
            ]  # Drift

        # Fill muID vector and momentum/chi2 arrays
        muID.clear()
        for pair_idx in range(max_pairs):
            muon_idx = pair_idx * 2
            muID.push_back(muon_idx + 1)  # μ⁺
            muID.push_back(muon_idx + 2)  # μ⁻

            qpx[muon_idx] = momentum_mup_all[i, pair_idx, 0]
            qpy[muon_idx] = momentum_mup_all[i, pair_idx, 1]
            qpz[muon_idx] = momentum_mup_all[i, pair_idx, 2]
            qchi2[muon_idx] = chi2_mup_all[i, pair_idx]

            qpx[muon_idx + 1] = momentum_mum_all[i, pair_idx, 0]
            qpy[muon_idx + 1] = momentum_mum_all[i, pair_idx, 1]
            qpz[muon_idx + 1] = momentum_mum_all[i, pair_idx, 2]
            qchi2[muon_idx + 1] = chi2_mum_all[i, pair_idx]

        output_tree.Fill()

    fout.Write()
    fout.Close()
    f_input.Close()
    print(f"Multi-pair predictions written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-track finder and momentum inference on ROOT file."
    )
    parser.add_argument("root_file", type=str, help="Path to input ROOT file.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="qtracker_multi_reco.root",
        help="Output ROOT file.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="Maximum number of dimuon pairs to reconstruct.",
    )
    args = parser.parse_args()

    process_data(args.root_file, args.output_file, args.max_pairs)
