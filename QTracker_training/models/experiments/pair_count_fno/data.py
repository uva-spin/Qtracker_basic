"""
Pair-count label derivation for the Axial-FNO classifier experiment.

Wraps the shared models/data_loader.load_data_denoise() rather than
duplicating ROOT-reading logic -- this experiment consumes the exact same
curriculum ROOT files (train_low/med/high, val) already used to train
MultiTrackFinder, just with a different derived label.
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np

_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from data_loader import load_data_denoise  # noqa: E402


def _derive_pair_count_labels(y_muPlus: np.ndarray, y_muMinus: np.ndarray) -> np.ndarray:
    """
    Args:
        y_muPlus: (N, P, 62) ground-truth mu+ element IDs per pair slot.
        y_muMinus: (N, P, 62) ground-truth mu- element IDs per pair slot.

    Returns:
        int32 array of shape (N,): count of occupied pair slots per event
        (a slot is occupied if either muon fires a non-zero element ID
        anywhere in that slot), i.e. the true number of dimuon pairs, 0..P.
    """
    occupied = np.any(y_muPlus != 0, axis=2) | np.any(y_muMinus != 0, axis=2)  # (N, P)
    return occupied.sum(axis=1).astype(np.int32)


def load_data_pair_count(
    root_file: str, max_pairs: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads noisy hit matrices and integer pair-count labels (0..max_pairs).

    Args:
        root_file: Path to the ROOT file (same curriculum files as MultiTrackFinder).
        max_pairs: Maximum number of dimuon pair slots recorded in the file.

    Returns:
        X: (N, 62, 201, 1) noisy hit matrices, or None on load failure.
        y_count: (N,) int32 pair-count labels, or None on load failure.
    """
    X, _X_clean, y_muPlus, y_muMinus = load_data_denoise(
        root_file, multi_track=True, max_pairs=max_pairs
    )
    if X is None:
        return None, None
    return X, _derive_pair_count_labels(y_muPlus, y_muMinus)
