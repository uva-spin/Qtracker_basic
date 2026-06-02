"""Feature extraction for per-track Quality Metric pattern studies."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .constants import (
    ACTIVE_MASK,
    BACK_DETECTORS,
    FRONT_DETECTORS,
    MIDDLE_DETECTORS,
    NO_HIT_ELEMENT_ID,
    NUM_ELEMENT_IDS,
    STATION_GROUPS,
    TAIL_DETECTORS,
)


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _active_indices(active_mask: np.ndarray) -> np.ndarray:
    return np.where(active_mask)[0]


def _count_hits(hit_array: np.ndarray, det_indices: Iterable[int]) -> int:
    indices = np.array(list(det_indices), dtype=np.int32)
    if indices.size == 0:
        return 0
    indices = indices[(indices >= 0) & (indices < hit_array.shape[0])]
    if indices.size == 0:
        return 0
    return int(np.count_nonzero(hit_array[indices] != NO_HIT_ELEMENT_ID))


def _max_consecutive_missing(hit_array: np.ndarray, active_mask: np.ndarray) -> int:
    active_hit_array = hit_array[active_mask]
    missing = active_hit_array == NO_HIT_ELEMENT_ID
    best_gap = 0
    current_gap = 0
    for is_missing in missing:
        if is_missing:
            current_gap += 1
            if current_gap > best_gap:
                best_gap = current_gap
        else:
            current_gap = 0
    return int(best_gap)


def _event_occupancy_features(event_hit_matrix: Optional[np.ndarray]) -> Dict[str, float]:
    if event_hit_matrix is None:
        return {
            "event_total_occupancy": np.nan,
            "event_active_occupancy": np.nan,
            "event_mean_layer_occupancy": np.nan,
            "event_max_layer_occupancy": np.nan,
        }

    matrix = np.asarray(event_hit_matrix)
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(
            f"event_hit_matrix must have shape (62, 201) or (62, 201, 1), got {matrix.shape}"
        )

    layer_counts = np.count_nonzero(matrix > 0, axis=1)
    active_layer_counts = layer_counts[ACTIVE_MASK]
    return {
        "event_total_occupancy": int(np.sum(layer_counts)),
        "event_active_occupancy": int(np.sum(active_layer_counts)),
        "event_mean_layer_occupancy": float(np.mean(active_layer_counts)) if active_layer_counts.size else 0.0,
        "event_max_layer_occupancy": int(np.max(active_layer_counts)) if active_layer_counts.size else 0,
    }


def _local_density_features(
    hit_array: np.ndarray,
    event_hit_matrix: Optional[np.ndarray],
    window: int,
    active_mask: np.ndarray,
) -> Dict[str, float]:
    if event_hit_matrix is None:
        return {
            "local_density_mean": np.nan,
            "local_density_max": np.nan,
        }

    matrix = np.asarray(event_hit_matrix)
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(
            f"event_hit_matrix must have shape (62, 201) or (62, 201, 1), got {matrix.shape}"
        )

    densities = []
    for det_idx, elem_id in enumerate(hit_array):
        if not active_mask[det_idx] or elem_id == NO_HIT_ELEMENT_ID:
            continue
        elem_id = int(elem_id)
        lo = max(0, elem_id - window)
        hi = min(matrix.shape[1], elem_id + window + 1)
        densities.append(float(np.count_nonzero(matrix[det_idx, lo:hi] > 0)))

    if not densities:
        return {
            "local_density_mean": 0.0,
            "local_density_max": 0.0,
        }
    return {
        "local_density_mean": float(np.mean(densities)),
        "local_density_max": float(np.max(densities)),
    }


def compute_candidate_features(
    hit_array: np.ndarray,
    event_hit_matrix: Optional[np.ndarray] = None,
    active_mask: np.ndarray = ACTIVE_MASK,
    local_window: int = 2,
) -> Dict[str, float]:
    """Compute observable, runtime-available features for one candidate track.

    Args:
        hit_array: Candidate element IDs with shape (62,). Element ID 0 is treated
            as no-hit, following the TrackFinder loss convention.
        event_hit_matrix: Optional event occupancy matrix with shape (62, 201) or
            (62, 201, 1). This is used only for occupancy/local-density features.
        active_mask: Boolean mask selecting detector slots used for scoring.
        local_window: Element-ID half-window for local density around selected hits.

    Returns:
        A flat dictionary of numeric features.
    """
    hit_array = np.asarray(hit_array, dtype=np.int32)
    if hit_array.shape != (active_mask.shape[0],):
        raise ValueError(f"hit_array must have shape ({active_mask.shape[0]},), got {hit_array.shape}")

    active_count = int(np.count_nonzero(active_mask))
    active_hits = hit_array[active_mask]
    hit_count = int(np.count_nonzero(active_hits != NO_HIT_ELEMENT_ID))
    missing_count = active_count - hit_count

    features: Dict[str, float] = {
        "hit_count_active": hit_count,
        "missing_count_active": missing_count,
        "missing_fraction_active": _safe_fraction(missing_count, active_count),
        "hit_fraction_active": _safe_fraction(hit_count, active_count),
        "max_missing_gap_active": _max_consecutive_missing(hit_array, active_mask),
    }

    for station_name, dets in STATION_GROUPS.items():
        station_dets = [det for det in dets if 0 <= det < len(active_mask) and active_mask[det]]
        station_size = len(station_dets)
        station_hits = _count_hits(hit_array, station_dets)
        features[f"{station_name}_hit_count"] = station_hits
        features[f"{station_name}_hit_fraction"] = _safe_fraction(station_hits, station_size)

    region_defs = {
        "front": FRONT_DETECTORS,
        "middle": MIDDLE_DETECTORS,
        "back": BACK_DETECTORS,
        "tail": TAIL_DETECTORS,
    }
    for region_name, dets in region_defs.items():
        region_dets = [det for det in dets if 0 <= det < len(active_mask) and active_mask[det]]
        region_size = len(region_dets)
        region_hits = _count_hits(hit_array, region_dets)
        features[f"{region_name}_hit_count"] = region_hits
        features[f"{region_name}_hit_fraction"] = _safe_fraction(region_hits, region_size)

    covered_stations = 0
    possible_stations = 0
    for station_name, dets in STATION_GROUPS.items():
        station_dets = [det for det in dets if 0 <= det < len(active_mask) and active_mask[det]]
        if not station_dets:
            continue
        possible_stations += 1
        if _count_hits(hit_array, station_dets) > 0:
            covered_stations += 1
    features["covered_station_count"] = covered_stations
    features["station_coverage_fraction"] = _safe_fraction(covered_stations, possible_stations)

    features.update(_event_occupancy_features(event_hit_matrix))
    features.update(_local_density_features(hit_array, event_hit_matrix, local_window, active_mask))

    return features


def compute_softmax_features(
    softmax: Optional[np.ndarray],
    hit_array: Optional[np.ndarray] = None,
    active_mask: np.ndarray = ACTIVE_MASK,
) -> Dict[str, float]:
    """Compute softmax-derived confidence features for one candidate track.

    Args:
        softmax: Array with shape (62, 201). If None, returns NaN features.
        hit_array: Optional argmax hit array. If not provided, argmax is computed.
        active_mask: Boolean mask selecting detector slots used for scoring.
    """
    nan_features = {
        "softmax_conf_mean": np.nan,
        "softmax_conf_min": np.nan,
        "softmax_margin_mean": np.nan,
        "softmax_entropy_mean": np.nan,
        "presence_prob_mean": np.nan,
        "presence_prob_min": np.nan,
    }
    if softmax is None:
        return nan_features

    probs = np.asarray(softmax, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"softmax must have shape (62, 201), got {probs.shape}")

    if hit_array is None:
        hit_array = np.argmax(probs, axis=-1).astype(np.int32)
    else:
        hit_array = np.asarray(hit_array, dtype=np.int32)

    active_probs = probs[active_mask]
    active_hits = hit_array[active_mask]
    if active_probs.size == 0:
        return nan_features

    row_indices = np.arange(active_probs.shape[0])
    selected_probs = active_probs[row_indices, active_hits]

    sorted_probs = np.sort(active_probs, axis=-1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if active_probs.shape[1] >= 2 else np.zeros_like(top1)
    margins = top1 - top2

    clipped = np.clip(active_probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1) / np.log(float(NUM_ELEMENT_IDS))

    presence = 1.0 - active_probs[:, NO_HIT_ELEMENT_ID]

    hit_detector_mask = active_hits != NO_HIT_ELEMENT_ID
    if np.any(hit_detector_mask):
        conf_for_hits = selected_probs[hit_detector_mask]
    else:
        conf_for_hits = selected_probs

    return {
        "softmax_conf_mean": float(np.mean(conf_for_hits)),
        "softmax_conf_min": float(np.min(conf_for_hits)),
        "softmax_margin_mean": float(np.mean(margins)),
        "softmax_entropy_mean": float(np.mean(entropy)),
        "presence_prob_mean": float(np.mean(presence)),
        "presence_prob_min": float(np.min(presence)),
    }
