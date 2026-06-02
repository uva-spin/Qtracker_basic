import ROOT
import numpy as np
from typing import Optional, Tuple


def load_data(
    root_file: str, multi_track: bool = False, max_pairs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a ROOT file and return input features and labels.

    Args:
        root_file (str): Path to the ROOT file.
        multi_track (bool): If True, load multi-track events. If False, load single-track events.
        max_pairs (Optional[int]): Maximum number of pairs to consider for multi-track events.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: Input features of shape (num_events, 62, 201, 1)
            - y_muPlus: Labels for muPlus of shape (num_events, 62) for single track, (num_events, max_pairs, 62) for multi-track
            - y_muMinus: Labels for muMinus of shape (num_events, 62) for single track, (num_events, max_pairs, 62) for multi-track
    """

    # Input validation
    if multi_track and max_pairs is None:
        raise ValueError("max_pairs must be provided when multi_track=True")
    if not multi_track and max_pairs is not None:
        raise ValueError("max_pairs should not be set when multi_track=False")

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

        mu_plus_array = np.array(list(event.HitArray_mup), dtype=np.int32)
        mu_minus_array = np.array(list(event.HitArray_mum), dtype=np.int32)

        if multi_track and max_pairs is not None:
            mu_plus_array = mu_plus_array.reshape(max_pairs, num_detectors)
            mu_minus_array = mu_minus_array.reshape(max_pairs, num_detectors)

        X.append(event_hits_matrix)
        y_muPlus.append(mu_plus_array)
        y_muMinus.append(mu_minus_array)

    X = np.array(X)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    y_muPlus = np.array(y_muPlus)
    y_muMinus = np.array(y_muMinus)

    return X, y_muPlus, y_muMinus


def load_data_denoise(
    root_file: str, multi_track: bool = False, max_pairs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a ROOT file for denoising tasks. It returns an additional
    np.ndarray for clean input features (no background tracks) to use as label for denoiser.

    Args:
        root_file (str): Path to the ROOT file.
        multi_track (bool): If True, load multi-track events. If False, load single-track events.
        max_pairs (Optional[int]): Maximum number of pairs to consider for multi-track events.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X: Noisy input features of shape (num_events, 62, 201, 1)
            - X_clean: Clean input features of shape (num_events, 62, 201, 1)
            - y_muPlus: Labels for muPlus of shape (num_events, 62) for single track, (num_events, max_pairs, 62) for multi-track
            - y_muMinus: Labels for muMinus of shape (num_events, 62) for single track, (num_events, max_pairs, 62) for multi-track
    """

    # Input validation
    if multi_track and max_pairs is None:
        raise ValueError("max_pairs must be provided when multi_track=True")
    if not multi_track and max_pairs is not None:
        raise ValueError("max_pairs should not be set when multi_track=False")

    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")

    if not tree:
        print("Error: Tree not found in file.")
        return None, None, None, None

    num_detectors = 62
    num_elementIDs = 201

    X = []
    X_clean = []
    y_muPlus = []
    y_muMinus = []

    for event in tree:
        event_hits_matrix = np.zeros((num_detectors, num_elementIDs), dtype=np.float32)
        clean_event_hits_matrix = np.zeros(
            (num_detectors, num_elementIDs), dtype=np.float32
        )

        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                event_hits_matrix[det_id, elem_id] = 1

        for det_id, elem_id in zip(event.detectorIDClean, event.elementIDClean):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                clean_event_hits_matrix[det_id, elem_id] = 1

        mu_plus_array = np.array(list(event.HitArray_mup), dtype=np.int32)
        mu_minus_array = np.array(list(event.HitArray_mum), dtype=np.int32)

        if multi_track and max_pairs is not None:
            mu_plus_array = mu_plus_array.reshape(max_pairs, num_detectors)
            mu_minus_array = mu_minus_array.reshape(max_pairs, num_detectors)

        if np.any(mu_plus_array < 0) or np.any(mu_plus_array >= num_elementIDs):
            continue

        if np.any(mu_minus_array < 0) or np.any(mu_minus_array >= num_elementIDs):
            continue

        X.append(event_hits_matrix)
        X_clean.append(clean_event_hits_matrix)
        y_muPlus.append(mu_plus_array)
        y_muMinus.append(mu_minus_array)

    X = np.array(X)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    X_clean = np.array(X_clean)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    y_muPlus = np.array(y_muPlus)
    y_muMinus = np.array(y_muMinus)

    return X, X_clean, y_muPlus, y_muMinus
