"""
data_loader_numpy.py — ROOT-free data loader for .npz files produced by data/root_to_numpy.py.

Drop-in replacement for load_data_denoise() when ROOT is not available
(Google Colab, local Mac, etc.).

Arrays in each .npz:
    X        : (N, 62, 201, 1)  float32 — noisy hit matrix
    X_clean  : (N, 62, 201, 1)  float32 — clean hit matrix (denoiser target)
    y_mup    : (N, max_pairs, 62) int8   — μ+ element-ID labels per pair
    y_mum    : (N, max_pairs, 62) int8   — μ- element-ID labels per pair
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def load_data_numpy(
    npz_path: str,
    max_pairs: Optional[int] = None,
    max_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .npz file produced by data/root_to_numpy.py.

    Args:
        npz_path:   Path to the .npz file (with or without the .npz extension).
        max_pairs:  If provided, assert that the stored max_pairs matches.
        max_events: If provided, only load the first N events (useful for local smoke tests).

    Returns:
        X        : (N, 62, 201, 1)  float32
        X_clean  : (N, 62, 201, 1)  float32
        y_mup    : (N, max_pairs, 62) float32
        y_mum    : (N, max_pairs, 62) float32
    """
    if not npz_path.endswith(".npz"):
        npz_path = npz_path + ".npz"

    data = np.load(npz_path)

    X: np.ndarray = data["X"]
    X_clean: np.ndarray = data["X_clean"]
    y_mup: np.ndarray = data["y_mup"].astype(np.float32)
    y_mum: np.ndarray = data["y_mum"].astype(np.float32)

    if max_pairs is not None and y_mup.shape[1] != max_pairs:
        raise ValueError(
            f"max_pairs mismatch: file has {y_mup.shape[1]}, expected {max_pairs}"
        )

    if max_events is not None:
        X = X[:max_events]
        X_clean = X_clean[:max_events]
        y_mup = y_mup[:max_events]
        y_mum = y_mum[:max_events]

    print(
        f"Loaded {X.shape[0]} events from {npz_path} "
        f"| X: {X.shape} X_clean: {X_clean.shape} "
        f"y_mup: {y_mup.shape} y_mum: {y_mum.shape}",
        flush=True,
    )
    return X, X_clean, y_mup, y_mum
