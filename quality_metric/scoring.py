"""Initial interpretable Quality Metric scoring prototype.

This is intentionally simple. It gives us a first thresholdable score while the
pattern table tells us which features actually correlate with reconstruction
quality. Higher score means better candidate quality.
"""

from __future__ import annotations

import math
from typing import Mapping


def _value(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(val):
        return default
    return val


def score_candidate(row: Mapping[str, float]) -> float:
    """Compute a conservative first-pass quality score in [0, 1]."""
    hit_fraction = _value(row, "hit_fraction_active")
    station_coverage = _value(row, "station_coverage_fraction")
    softmax_conf = _value(row, "softmax_conf_mean", 0.5)
    margin = _value(row, "softmax_margin_mean", 0.0)
    entropy = _value(row, "softmax_entropy_mean", 1.0)
    missing_gap = _value(row, "max_missing_gap_active", 0.0)
    occupancy = _value(row, "event_mean_layer_occupancy", 0.0)

    gap_penalty = min(1.0, missing_gap / 12.0)
    occupancy_penalty = min(1.0, occupancy / 20.0)

    score = (
        0.30 * hit_fraction
        + 0.25 * station_coverage
        + 0.20 * softmax_conf
        + 0.10 * margin
        + 0.10 * (1.0 - entropy)
        + 0.05 * (1.0 - gap_penalty)
        - 0.10 * occupancy_penalty
    )
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)
