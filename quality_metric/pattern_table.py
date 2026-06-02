"""Build candidate-level pattern tables from TrackFinder outputs."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .candidate_adapter import predictions_to_candidates
from .features import compute_candidate_features, compute_softmax_features
from .truth_compare import compare_to_truth


def build_pattern_rows(
    predictions,
    y_mup: Optional[np.ndarray] = None,
    y_mum: Optional[np.ndarray] = None,
    event_hit_matrices: Optional[np.ndarray] = None,
    min_candidate_hits: int = 0,
) -> List[dict]:
    rows = []
    for candidate in predictions_to_candidates(
        predictions,
        y_mup=y_mup,
        y_mum=y_mum,
        event_hit_matrices=event_hit_matrices,
        min_candidate_hits=min_candidate_hits,
    ):
        hit_array = candidate["hit_array"]
        softmax = candidate["softmax"]
        truth_hit_array = candidate["truth_hit_array"]
        event_hit_matrix = candidate["event_hit_matrix"]

        row = {
            "event_id": int(candidate["event_id"]),
            "pair_index": int(candidate["pair_index"]),
            "charge_index": int(candidate["charge_index"]),
            "charge": candidate["charge"],
        }
        row.update(compute_candidate_features(hit_array, event_hit_matrix=event_hit_matrix))
        row.update(compute_softmax_features(softmax, hit_array=hit_array))
        row.update(compare_to_truth(hit_array, truth_hit_array))
        rows.append(row)
    return rows
