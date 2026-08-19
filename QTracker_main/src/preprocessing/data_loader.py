import numpy as np
import os
import uproot


def load_data_without_labels(root_file: str) -> np.ndarray:
    """
    Load data without labels from a ROOT file using uproot and return input features.

    Args:
        root_file (str): Path to the ROOT file.
    Returns:
        np.ndarray: Input features of shape (num_events, 62, 201, 1)
    """
    num_detectors = 62
    num_elementIDs = 201

    with uproot.open(root_file) as f:
        tree = f["tree"]

        detectorIDs = tree["detectorID"].array(library="np")
        elementIDs = tree["elementID"].array(library="np")

    num_events = len(detectorIDs)

    X = np.zeros((num_events, num_detectors, num_elementIDs, 1), dtype=np.float32)

    for i in range(num_events):
        det_id = detectorIDs[i]
        elem_id = elementIDs[i]

        hit = (
            (det_id >= 0)
            & (det_id < num_detectors)
            & (elem_id >= 0)
            & (elem_id < num_elementIDs)
        )
        X[i, det_id[hit], elem_id[hit], 0] = 1.0

    return X


def load_data_with_labels(root_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data with labels from a ROOT file using uproot and return input features and labels.

    Args:
        root_file (str): Path to the ROOT file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: Input features of shape (num_events, 62, 201, 1)
            - y_muPlus: Labels for muPlus of shape (num_events, 201)
            - y_muMinus: Labels for muMinus of shape (num_events, 201)
    """
    num_detectors = 62
    num_elementIDs = 201

    with uproot.open(root_file) as f:
        tree = f["tree"]

        detectorIDs = tree["detectorID"].array(library="np")
        elementIDs = tree["elementID"].array(library="np")
        hitArray_mup = tree["HitArray_mup"].array(library="np")
        hitArray_mum = tree["HitArray_mum"].array(library="np")

    num_events = len(detectorIDs)

    X = np.zeros((num_events, num_detectors, num_elementIDs, 1), dtype=np.float32)
    y_muPlus = np.asarray(hitArray_mup, dtype=np.int32)
    y_muMinus = np.asarray(hitArray_mum, dtype=np.int32)

    for i in range(num_events):
        det_id = detectorIDs[i]
        elem_id = elementIDs[i]

        hit = (
            (det_id >= 0)
            & (det_id < num_detectors)
            & (elem_id >= 0)
            & (elem_id < num_elementIDs)
        )
        X[i, det_id[hit], elem_id[hit], 0] = 1.0

    return X, y_muPlus, y_muMinus


def load_detector_element_data(
    root_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads detectorID, elementID, driftDistance, and tdcTime from ROOT file.

    Args:
        root_file (str): Path to the ROOT file.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays of detectorIDs, elementIDs, driftDistances, and tdcTimes.
    """
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Error: Input ROOT file '{root_file}' not found.")

    detectorIDs, elementIDs, driftDistances, tdcTimes = [], [], [], []
    with uproot.open(root_file) as f:
        if "tree" not in f:
            raise ValueError(f"Error: 'tree' not found in {root_file}.")

        tree = f["tree"]

        detectorIDs = tree["detectorID"].array(library="np").astype(np.int32)
        elementIDs = tree["elementID"].array(library="np").astype(np.int32)
        driftDistances = tree["driftDistance"].array(library="np").astype(np.float32)
        tdcTimes = tree["tdcTime"].array(library="np").astype(np.float32)

    return detectorIDs, elementIDs, driftDistances, tdcTimes
