"""
Autoregressive Multi-Track Finder.

Uses a trained single-track TrackFinder iteratively:
1. Predict the highest-confidence dimuon pair.
2. Remove predicted hits from the input matrix.
3. Repeat until confidence drops below threshold or max_pairs is reached.
"""

# ruff: noqa: E402

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
from layers import AxialAttention
from data_loader import load_data

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201
INFERENCE_CHUNK_SIZE = 128


def compute_confidence(softmax_mup, softmax_mum):
    """
    Compute a confidence score for the predicted pair.

    Uses the mean max-probability across all detectors for both mu+ and mu-.
    High confidence means the model is certain about its element ID picks.
    Low confidence (close to 1/201 ≈ 0.005) means the model is guessing.

    Args:
        softmax_mup: (num_events, 62, 201) softmax output for mu+
        softmax_mum: (num_events, 62, 201) softmax output for mu-

    Returns:
        confidence: (num_events,) scalar confidence per event
    """
    max_prob_mup = np.max(softmax_mup, axis=-1)  # (num_events, 62)
    max_prob_mum = np.max(softmax_mum, axis=-1)  # (num_events, 62)

    # Average across detectors (excluding masked detectors 7-12, 55-62)
    mask = np.ones(62, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    conf_mup = np.mean(max_prob_mup[:, mask], axis=-1)  # (num_events,)
    conf_mum = np.mean(max_prob_mum[:, mask], axis=-1)

    return (conf_mup + conf_mum) / 2.0


def remove_predicted_hits(hit_matrix, hit_array_mup, hit_array_mum):
    """
    Remove predicted hits from the input hit matrix.

    For each detector where the model predicted a nonzero element ID,
    set that position in the hit matrix to 0.

    Args:
        hit_matrix: (num_events, 62, 201, 1) binary hit matrix
        hit_array_mup: (num_events, 62) predicted mu+ element IDs
        hit_array_mum: (num_events, 62) predicted mu- element IDs

    Returns:
        hit_matrix_residual: (num_events, 62, 201, 1) with predicted hits removed
    """
    residual = hit_matrix.copy()
    num_events = hit_matrix.shape[0]

    for i in range(num_events):
        for det in range(NUM_DETECTORS):
            elem_p = hit_array_mup[i, det]
            elem_m = hit_array_mum[i, det]
            if 0 < elem_p < NUM_ELEMENT_IDS:
                residual[i, det, elem_p, 0] = 0
            if 0 < elem_m < NUM_ELEMENT_IDS:
                residual[i, det, elem_m, 0] = 0

    return residual


def predict_single_pass(model, X):
    """
    Run the single-track model on input hit matrices.

    Args:
        model: loaded single-track TrackFinder model
        X: (num_events, 62, 201, 1) input hit matrices

    Returns:
        hit_array_mup: (num_events, 62) predicted mu+ element IDs
        hit_array_mum: (num_events, 62) predicted mu- element IDs
        softmax_mup: (num_events, 62, 201) softmax probabilities for mu+
        softmax_mum: (num_events, 62, 201) softmax probabilities for mu-
    """
    preds = []

    for i in range(0, len(X), INFERENCE_CHUNK_SIZE):
        X_chunk = tf.cast(X[i : i + INFERENCE_CHUNK_SIZE], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        preds.append(y_chunk[1])  # segment output

    predictions = np.concatenate(preds, axis=0)

    softmax_mup = predictions[:, 0, :, :]  # (num_events, 62, 201)
    softmax_mum = predictions[:, 1, :, :]
    hit_array_mup = np.argmax(softmax_mup, axis=-1).astype(np.int32)
    hit_array_mum = np.argmax(softmax_mum, axis=-1).astype(np.int32)

    return hit_array_mup, hit_array_mum, softmax_mup, softmax_mum


def autoregressive_predict(model, X, max_pairs=5, confidence_threshold=0.5):
    """
    Iterative multi-track prediction.

    For each event, repeatedly run the single-track model:
    1. Predict one pair from the current hit matrix.
    2. Compute confidence. If below threshold, stop for this event.
    3. Remove predicted hits from the hit matrix.
    4. Repeat up to max_pairs times.

    Args:
        model: loaded single-track TrackFinder model
        X: (num_events, 62, 201, 1) input hit matrices
        max_pairs: maximum dimuon pairs to extract (and GT slots in ROOT data)
        confidence_threshold: stop when confidence drops below this

    Returns:
        all_pairs_mup: list of (num_events, 62) arrays, one per iteration
        all_pairs_mum: list of (num_events, 62) arrays, one per iteration
        all_confidences: list of (num_events,) arrays, one per iteration
        n_pairs_per_event: (num_events,) number of valid pairs found per event
    """
    num_events = X.shape[0]
    residual = X.copy()

    all_pairs_mup = []
    all_pairs_mum = []
    all_confidences = []

    # Track which events are still active (haven't stopped)
    active = np.ones(num_events, dtype=bool)
    n_pairs_per_event = np.zeros(num_events, dtype=np.int32)

    for iteration in range(max_pairs):
        print(f"  Pair {iteration + 1}/{max_pairs} — {np.sum(active)} active events")

        if not np.any(active):
            break

        # Only run inference on active events
        active_idx = np.where(active)[0]

        a_mup, a_mum, a_smax_mup, a_smax_mum = predict_single_pass(
            model, residual[active_idx]
        )
        a_conf = compute_confidence(a_smax_mup, a_smax_mum)

        passes_threshold = a_conf >= confidence_threshold

        # Full-size output arrays (zeros for inactive events)
        mup = np.zeros((num_events, NUM_DETECTORS), dtype=np.int32)
        mum = np.zeros((num_events, NUM_DETECTORS), dtype=np.int32)
        confidence = np.zeros(num_events)

        newly_active_idx = active_idx[passes_threshold]
        mup[newly_active_idx] = a_mup[passes_threshold]
        mum[newly_active_idx] = a_mum[passes_threshold]
        confidence[active_idx] = a_conf

        all_pairs_mup.append(mup.copy())
        all_pairs_mum.append(mum.copy())
        all_confidences.append(confidence.copy())

        # Update pair counts for newly found pairs
        n_pairs_per_event[newly_active_idx] += 1

        # Remove predicted hits for active events
        residual = remove_predicted_hits(residual, mup, mum)

        # Update active mask
        active = np.zeros(num_events, dtype=bool)
        active[newly_active_idx] = True

    return all_pairs_mup, all_pairs_mum, all_confidences, n_pairs_per_event


def main(args):
    """Main entry point for autoregressive multi-track finding."""

    # Load data
    print(f"Loading data from {args.root_file}...")
    X, y_muPlus, y_muMinus = load_data(
        args.root_file, multi_track=True, max_pairs=args.max_pairs
    )
    if X is None:
        print("Error loading data.")
        return

    print(f"Loaded {len(X)} events.")

    # Load model
    print(f"Loading model from {args.model_path}...")
    custom_objects = {"AxialAttention": AxialAttention}
    model = tf.keras.models.load_model(
        args.model_path, compile=False, custom_objects=custom_objects
    )

    # Run autoregressive prediction
    print(
        f"Running autoregressive prediction (max_pairs={args.max_pairs}, "
        f"threshold={args.confidence_threshold})..."
    )
    all_mup, all_mum, all_conf, n_pairs = autoregressive_predict(
        model,
        X,
        max_pairs=args.max_pairs,
        confidence_threshold=args.confidence_threshold,
    )

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("Autoregressive Prediction Summary")
    print(f"{'=' * 60}")
    for k in range(1, args.max_pairs + 1):
        count = np.sum(n_pairs >= k)
        print(
            f"  Events with >= {k} pairs found: {count} ({100 * count / len(X):.1f}%)"
        )

    # Save predictions to a numpy file for downstream evaluation
    output_path = args.output if args.output else "autoregressive_predictions.npz"
    np.savez_compressed(
        output_path,
        all_mup=np.array(all_mup),  # (max_pairs, num_events, 62)
        all_mum=np.array(all_mum),
        all_conf=np.array(all_conf),  # (max_pairs, num_events)
        n_pairs=n_pairs,  # (num_events,)
        y_muPlus=y_muPlus,  # GT
        y_muMinus=y_muMinus,
    )
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoregressive multi-track finding using iterative single-track prediction."
    )
    parser.add_argument("root_file", type=str, help="Path to multi-track ROOT file.")
    parser.add_argument(
        "model_path", type=str, help="Path to trained single-track model."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.15,
        help="Stop when mean softmax confidence drops below this value.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="Max dimuon pairs per event (extraction limit and ROOT GT layout).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz file path for predictions.",
    )
    args = parser.parse_args()
    main(args)
