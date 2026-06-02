"""Adapters from TrackFinder/MultiTrackFinder outputs to candidate rows."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

import numpy as np

from .constants import NUM_DETECTORS, NUM_ELEMENT_IDS

CHARGE_LABELS = {
    0: "mup",
    1: "mum",
}


def normalize_segment_output(predictions) -> np.ndarray:
    """Return the segmentation output from a model prediction object.

    Supports both:
        - direct segmentation output: (N, 2, 62, 201) or (N, P, 2, 62, 201)
        - denoiser + segmentation output: [denoise_out, seg_output]
    """
    if isinstance(predictions, (list, tuple)):
        if len(predictions) < 2:
            raise ValueError("Prediction list/tuple must contain a segmentation output.")
        segment = predictions[-1]
    else:
        segment = predictions

    segment = np.asarray(segment)
    if segment.ndim == 4:
        expected = (2, NUM_DETECTORS, NUM_ELEMENT_IDS)
        if segment.shape[1:] != expected:
            raise ValueError(
                f"Single-track segment output must have shape (N, 2, 62, 201), got {segment.shape}"
            )
        return segment[:, np.newaxis, :, :, :]

    if segment.ndim == 5:
        expected_tail = (2, NUM_DETECTORS, NUM_ELEMENT_IDS)
        if segment.shape[2:] != expected_tail:
            raise ValueError(
                f"Multi-track segment output must have shape (N, P, 2, 62, 201), got {segment.shape}"
            )
        return segment

    raise ValueError(
        f"Unsupported segment output rank {segment.ndim}. Expected rank 4 or 5, got shape {segment.shape}."
    )


def _truth_for_candidate(
    y_mup: Optional[np.ndarray],
    y_mum: Optional[np.ndarray],
    event_idx: int,
    pair_idx: int,
    charge_idx: int,
) -> Optional[np.ndarray]:
    if y_mup is None or y_mum is None:
        return None
    y = y_mup if charge_idx == 0 else y_mum
    y = np.asarray(y)
    if y.ndim == 2:
        if pair_idx != 0:
            return None
        return y[event_idx].astype(np.int32)
    if y.ndim == 3:
        return y[event_idx, pair_idx].astype(np.int32)
    raise ValueError(f"Truth array must have rank 2 or 3, got {y.shape}")


def predictions_to_candidates(
    predictions,
    y_mup: Optional[np.ndarray] = None,
    y_mum: Optional[np.ndarray] = None,
    event_hit_matrices: Optional[np.ndarray] = None,
    min_candidate_hits: int = 0,
) -> Iterator[Dict[str, object]]:
    """Yield candidate dictionaries from model predictions.

    The returned dicts are intentionally lightweight. Heavy arrays are included so
    downstream feature extraction can decide what to keep.
    """
    segment = normalize_segment_output(predictions)
    num_events, max_pairs = segment.shape[0], segment.shape[1]

    for event_idx in range(num_events):
        event_hit_matrix = None
        if event_hit_matrices is not None:
            event_hit_matrix = event_hit_matrices[event_idx]
        for pair_idx in range(max_pairs):
            for charge_idx in range(2):
                softmax = segment[event_idx, pair_idx, charge_idx]
                hit_array = np.argmax(softmax, axis=-1).astype(np.int32)
                hit_count = int(np.count_nonzero(hit_array != 0))
                if hit_count < min_candidate_hits:
                    continue

                yield {
                    "event_id": event_idx,
                    "pair_index": pair_idx,
                    "charge_index": charge_idx,
                    "charge": CHARGE_LABELS[charge_idx],
                    "hit_array": hit_array,
                    "softmax": softmax,
                    "truth_hit_array": _truth_for_candidate(
                        y_mup, y_mum, event_idx, pair_idx, charge_idx
                    ),
                    "event_hit_matrix": event_hit_matrix,
                }
