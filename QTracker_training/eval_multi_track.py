# ruff: noqa: E402

import os
import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity("error")

import argparse
import itertools
import ROOT  # noqa: F401
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# core TrackFinder loaders / custom loss
from models import data_loader
from models.layers import AxialAttention
import QTracker
from refine import refine_hit_arrays


def plot_residuals(det_ids, res_plus, res_minus, model_path, stage_label):
    mean_p = np.nanmean(np.abs(res_plus), axis=0)
    std_p = np.nanstd(np.abs(res_plus), axis=0)
    mean_m = np.nanmean(np.abs(res_minus), axis=0)
    std_m = np.nanstd(np.abs(res_minus), axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(det_ids, mean_p, yerr=std_p, marker="o", label="μ+ mean±σ")
    plt.errorbar(det_ids, mean_m, yerr=std_m, marker="s", label="μ- mean±σ")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Detector Layer (skipping masked slots)")
    plt.ylabel("Absolute Residual (predicted − true)")
    plt.title(f"Per-layer Absolute Residual ({stage_label.capitalize()})")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(model_path))[0]
    fname = f"{base}_{stage_label}_residuals.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots", "multi_track")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()


def chi_squared(y_true, y_pred):
    # y_true.shape = (num_events, 62)
    # y_pred.shape = (num_events, 62)
    residuals = y_true - y_pred
    sigma = np.std(y_true, axis=0) + 1e-6  # Prevent division by zero

    res_norm = residuals / sigma

    chi2 = np.sum((res_norm**2), axis=1)  # Chi-squared per event
    chi2_mean = np.mean(chi2)  # Mean chi-squared over all events
    return chi2_mean


def match_predictions(y_test, y_pred_argmax, mask):
    """Reorder predicted pair slots per event to best match the active GT pairs.

    For each event we restrict to the active GT pairs (slots with any nonzero
    hit; GT is assumed already canonically sorted). We build a residual-based
    cost between every predicted slot ``p`` and every active GT slot ``g``:

        cost(p, g) = sum_{unmasked det} |pred_p - gt_g| for mu+
                   + sum_{unmasked det} |pred_p - gt_g| for mu-

    We enumerate permutations of predicted slots and pick the one minimizing
    the total cost over the active GT slots. The chosen permutation is applied
    so the predicted slot matched to GT rank ``k`` lands at index ``k``.
    GT and model pair-slot counts may differ (e.g. 3-pair GT vs 5-pair model).
    """
    matched = y_pred_argmax.copy()
    det_idx = np.where(mask)[0]

    for ev in range(len(y_test)):
        gt = y_test[ev].astype(np.int32)  # (n_gt, 2, 62)
        pred = y_pred_argmax[ev]  # (n_pred, 2, 62)
        n_gt = gt.shape[0]
        n_pred = pred.shape[0]

        n_active = max(
            (k + 1 for k in range(n_gt) if np.any(gt[k] != 0)),
            default=0,
        )
        if n_active <= 1:
            continue

        n_match = min(n_active, n_pred)

        cost = np.zeros((n_pred, n_match))
        for p in range(n_pred):
            for g in range(n_match):
                c_plus = np.sum(np.abs(pred[p, 0, det_idx] - gt[g, 0, det_idx]))
                c_minus = np.sum(np.abs(pred[p, 1, det_idx] - gt[g, 1, det_idx]))
                cost[p, g] = c_plus + c_minus

        best_perm = None
        best_cost = None
        for perm in itertools.permutations(range(n_pred), n_match):
            total = sum(cost[perm[k], k] for k in range(n_match))
            if best_cost is None or total < best_cost:
                best_cost = total
                best_perm = perm

        reordered = pred.copy()
        for k in range(n_match):
            reordered[k] = pred[best_perm[k]]
        matched[ev] = reordered

    return matched


def evaluate_model(args):
    # Load data - existing loader handles both formats
    load_result = data_loader.load_data(
        args.root_file, multi_track=True, max_pairs=args.max_pairs
    )
    X_test, y_muPlus_test, y_muMinus_test = load_result[:3]

    if X_test is None:
        return

    # Detect format from shape
    is_multi_track = len(y_muPlus_test.shape) == 3  # (num_events, max_pairs, 62)

    if is_multi_track:
        num_events, max_pairs, num_detectors = y_muPlus_test.shape
        print(f"\n{'=' * 70}")
        print(f"Multi-track format detected: max_pairs={max_pairs}")
        print(f"{'=' * 70}\n")
        y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=2)
        # Shape: (num_events, max_pairs, 2, 62)

        # Apply canonical ordering to GT pairs so evaluation matches training ordering
        for ev_idx in range(len(y_test)):
            n_active = max(
                (
                    k + 1
                    for k in range(y_test.shape[1])
                    if np.any(y_test[ev_idx, k] != 0)
                ),
                default=0,
            )
            if n_active > 1:
                active_indices = np.arange(n_active)
                sort_keys = [
                    np.sum(y_test[ev_idx, k, 0, :][y_test[ev_idx, k, 0, :] > 0])
                    for k in active_indices
                ]
                sorted_order = active_indices[np.argsort(sort_keys)]
                y_test[ev_idx, :n_active] = y_test[ev_idx, sorted_order]
    else:
        print("\n" + "=" * 70)
        print("Single-track format detected")
        print("=" * 70 + "\n")
        y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=1)
        # Reshape to (num_events, 1, 2, 62) for uniform processing
        y_test = y_test[:, np.newaxis, :, :]
        max_pairs = 1

    det_test, elem_test, _, _, _ = QTracker.load_detector_element_data(args.root_file)

    mask = np.ones(62, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    custom_objects = {"AxialAttention": AxialAttention}
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects=custom_objects,
    )

    # Run predictions
    preds = []
    chunk_size = 128

    for i in range(0, len(X_test), chunk_size):
        X_chunk = tf.cast(X_test[i : i + chunk_size], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        preds.append(y_chunk[1])  # Segment output

    y_pred = np.concatenate(preds, axis=0)

    # Check prediction shape to determine format
    if len(y_pred.shape) == 5:  # Multi-track: (num_events, max_pairs, 2, 62, 201)
        print("Multi-track model predictions")
    elif len(y_pred.shape) == 4:  # Single-track: (num_events, 2, 62, 201)
        print("Single-track model predictions - reshaping for uniform processing\n")
        y_pred = y_pred[:, np.newaxis, :, :, :]  # Add pair dimension

    # Extract argmax predictions
    y_pred_argmax = np.argmax(y_pred, axis=-1).astype(np.int32)
    # Shape: (num_events, pred_max_pairs, 2, 62)

    gt_max_pairs = y_test.shape[1]
    pred_max_pairs = y_pred_argmax.shape[1]
    if gt_max_pairs != pred_max_pairs:
        print(
            f"WARNING: GT has {gt_max_pairs} pair slots but model outputs "
            f"{pred_max_pairs}. Evaluating the first "
            f"{min(gt_max_pairs, pred_max_pairs)} pair(s)."
        )
    eval_max_pairs = min(gt_max_pairs, pred_max_pairs)

    # Reorder predicted pair slots per event to best match the active GT pairs
    y_pred_argmax = match_predictions(y_test, y_pred_argmax, mask)

    # Evaluate each pair
    for pair_idx in range(eval_max_pairs):
        print(f"\n{'=' * 70}")
        print(f"Evaluating Pair {pair_idx}")
        print(f"{'=' * 70}")

        # Restrict to events where this GT rank is active (any nonzero hit)
        valid_mask = np.any(y_test[:, pair_idx, :, :] != 0, axis=(1, 2))

        num_valid = np.sum(valid_mask)
        if num_valid == 0:
            print(f"No valid events for pair {pair_idx}, skipping...")
            continue

        print(f"Valid events: {num_valid}/{len(y_test)}")

        # Extract matched predictions and ground truth for this pair
        y_p_raw = y_pred_argmax[valid_mask, pair_idx, 0, :]  # (valid_events, 62)
        y_m_raw = y_pred_argmax[valid_mask, pair_idx, 1, :]

        y_p_true = y_test[valid_mask, pair_idx, 0, :].astype(np.int32)
        y_m_true = y_test[valid_mask, pair_idx, 1, :].astype(np.int32)

        # Compute raw residuals (SAME as single-track)
        raw_p_res = y_p_true - y_p_raw
        raw_m_res = y_m_true - y_m_raw

        print("\n--- Raw Residuals (Before Refinement) ---")
        print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        for det in np.where(mask)[0]:
            m_p = np.mean(np.abs(raw_p_res[:, det]))
            s_p = np.std(np.abs(raw_p_res[:, det]))
            m_m = np.mean(np.abs(raw_m_res[:, det]))
            s_m = np.std(np.abs(raw_m_res[:, det]))
            print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

        # Plot residuals with pair index in filename
        dets_used = np.where(mask)[0] + 1
        plot_residuals(
            dets_used,
            raw_p_res[:, mask],
            raw_m_res[:, mask],
            args.model_path,
            f"pair{pair_idx}_raw",
        )

        # Refinement (filter detector data for valid events)
        det_valid = [det_test[i] for i in np.where(valid_mask)[0]]
        elem_valid = [elem_test[i] for i in np.where(valid_mask)[0]]

        ref_p, ref_m = refine_hit_arrays(y_p_raw, y_m_raw, det_valid, elem_valid)
        ref_p_res = y_p_true - ref_p
        ref_m_res = y_m_true - ref_m

        print("\n--- Refined Residuals (After Refinement) ---")
        print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        for det in np.where(mask)[0]:
            m_p = np.mean(np.abs(ref_p_res[:, det]))
            s_p = np.std(np.abs(ref_p_res[:, det]))
            m_m = np.mean(np.abs(ref_m_res[:, det]))
            s_m = np.std(np.abs(ref_m_res[:, det]))
            print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

        # Plot refined residuals
        plot_residuals(
            dets_used,
            ref_p_res[:, mask],
            ref_m_res[:, mask],
            args.model_path,
            f"pair{pair_idx}_refined",
        )

        # Calculate accuracy and chi-squared (SAME metrics as single-track)
        acc_p = np.mean(np.abs(raw_p_res) == 0)
        acc_m = np.mean(np.abs(raw_m_res) == 0)
        print(f"\nRaw μ+ accuracy: {acc_p:.4f}")
        print(f"Raw μ- accuracy: {acc_m:.4f}")

        acc_p = np.mean(np.abs(raw_p_res) <= 2)
        acc_m = np.mean(np.abs(raw_m_res) <= 2)
        print(f"\nRaw μ+ within-2 accuracy: {acc_p:.4f}")
        print(f"Raw μ- within-2 accuracy: {acc_m:.4f}")

        chi2_p = chi_squared(y_p_true, y_p_raw)
        chi2_m = chi_squared(y_m_true, y_m_raw)
        print(f"\nRaw μ+ Chi-squared: {chi2_p:.3f}")
        print(f"Raw μ- Chi-squared: {chi2_m:.3f}")

        # Calculate accuracy and chi-squared after refinement
        acc_p = np.mean(np.abs(ref_p_res) == 0)
        acc_m = np.mean(np.abs(ref_m_res) == 0)
        print(f"\nRefined μ+ accuracy: {acc_p:.4f}")
        print(f"Refined μ- accuracy: {acc_m:.4f}")

        acc_p = np.mean(np.abs(ref_p_res) <= 2)
        acc_m = np.mean(np.abs(ref_m_res) <= 2)
        print(f"\nRefined μ+ within-2 accuracy: {acc_p:.4f}")
        print(f"Refined μ- within-2 accuracy: {acc_m:.4f}")

        chi2_p = chi_squared(y_p_true, ref_p)
        chi2_m = chi_squared(y_m_true, ref_m)
        print(f"\nRefined μ+ Chi-squared: {chi2_p:.3f}")
        print(f"Refined μ- Chi-squared: {chi2_m:.3f}")

        print("\n--- Raw Absolute Residuals (Before Refinement) ---")
        print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        m_p, s_p = np.mean(np.abs(raw_p_res)), np.std(np.abs(raw_p_res))
        m_m, s_m = np.mean(np.abs(raw_m_res)), np.std(np.abs(raw_m_res))
        print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

        print("\n--- Refined Absolute Residuals (After Refinement) ---")
        print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        m_p, s_p = np.mean(np.abs(ref_p_res)), np.std(np.abs(ref_p_res))
        m_m, s_m = np.mean(np.abs(ref_m_res)), np.std(np.abs(ref_m_res))
        print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models (single or multi-track)."
    )
    parser.add_argument("root_file", type=str, help="Path to the val/test ROOT file.")
    parser.add_argument(
        "model_path", type=str, help="Path to the saved model file (.h5 or .keras)."
    )
    parser.add_argument(
        "--batch_norm",
        type=int,
        default=0,
        help="Flag to set batch normalization: [0 = False, 1 = True].",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=64,
        help="Flag to set batch normalization: [0 = False, 1 = True].",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="Maximum number of dimuon pairs (for multi-track files).",
    )
    args = parser.parse_args()

    print(f"\nResults for {args.model_path}...")
    evaluate_model(args)
