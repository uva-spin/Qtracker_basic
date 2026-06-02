"""Calibration-only truth comparison helpers for Quality Metric studies."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .constants import ACTIVE_MASK, NO_HIT_ELEMENT_ID


def compare_to_truth(
    hit_array: np.ndarray,
    truth_hit_array: Optional[np.ndarray],
    active_mask: np.ndarray = ACTIVE_MASK,
) -> Dict[str, float]:
    """Compare a candidate hit array to truth labels.

    These outputs must be used only for pattern-study calibration, not as runtime
    QMetric inputs.
    """
    if truth_hit_array is None:
        return {
            "truth_hit_count_active": np.nan,
            "pred_hit_count_active": np.nan,
            "exact_match_count": np.nan,
            "residual_leq_1_count": np.nan,
            "residual_leq_2_count": np.nan,
            "mean_abs_residual_on_truth_hits": np.nan,
            "max_abs_residual_on_truth_hits": np.nan,
            "missing_truth_hit_count": np.nan,
            "extra_pred_hit_count": np.nan,
        }

    pred = np.asarray(hit_array, dtype=np.int32)
    truth = np.asarray(truth_hit_array, dtype=np.int32)
    if pred.shape != truth.shape:
        raise ValueError(f"pred and truth shapes must match, got {pred.shape} and {truth.shape}")

    pred = pred[active_mask]
    truth = truth[active_mask]

    pred_has_hit = pred != NO_HIT_ELEMENT_ID
    truth_has_hit = truth != NO_HIT_ELEMENT_ID
    both_have_hit = pred_has_hit & truth_has_hit

    residual = pred.astype(np.int32) - truth.astype(np.int32)
    abs_residual = np.abs(residual[both_have_hit])

    if abs_residual.size > 0:
        mean_abs_residual = float(np.mean(abs_residual))
        max_abs_residual = float(np.max(abs_residual))
    else:
        mean_abs_residual = np.nan
        max_abs_residual = np.nan

    return {
        "truth_hit_count_active": int(np.count_nonzero(truth_has_hit)),
        "pred_hit_count_active": int(np.count_nonzero(pred_has_hit)),
        "exact_match_count": int(np.count_nonzero((residual == 0) & truth_has_hit & pred_has_hit)),
        "residual_leq_1_count": int(np.count_nonzero((np.abs(residual) <= 1) & truth_has_hit & pred_has_hit)),
        "residual_leq_2_count": int(np.count_nonzero((np.abs(residual) <= 2) & truth_has_hit & pred_has_hit)),
        "mean_abs_residual_on_truth_hits": mean_abs_residual,
        "max_abs_residual_on_truth_hits": max_abs_residual,
        "missing_truth_hit_count": int(np.count_nonzero(truth_has_hit & ~pred_has_hit)),
        "extra_pred_hit_count": int(np.count_nonzero(pred_has_hit & ~truth_has_hit)),
    }
