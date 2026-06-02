"""Candidate-level Quality Metric utilities for QTracker.

This package is intentionally independent from QTracker.py. It is meant to run
on direct TrackFinder / MultiTrackFinder evaluation outputs first.
"""

from .constants import NUM_DETECTORS, NUM_ELEMENT_IDS, ACTIVE_MASK
from .features import compute_candidate_features, compute_softmax_features
from .candidate_adapter import predictions_to_candidates

__all__ = [
    "NUM_DETECTORS",
    "NUM_ELEMENT_IDS",
    "ACTIVE_MASK",
    "compute_candidate_features",
    "compute_softmax_features",
    "predictions_to_candidates",
]
