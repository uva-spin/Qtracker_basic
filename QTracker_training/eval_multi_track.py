# ruff: noqa: E402

import os
import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity("error")

import argparse
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
            n_active = 0
            for k in range(y_test.shape[1]):
                if np.any(y_test[ev_idx, k] != 0):
                    n_active = k + 1
                else:
                    break
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
    # Shape: (num_events, max_pairs, 2, 62)

    # Evaluate each pair
    for pair_idx in range(max_pairs):
        print(f"\n{'=' * 70}")
        print(f"Evaluating Pair {pair_idx}")
        print(f"{'=' * 70}")

        # ============================================================
        # Pair Existence Evaluation (captures FP and FN)
        # ============================================================

        print("\n--- Pair Existence Metrics ---")

        # Ground truth existence
        gt_exists = np.any(y_test[:, pair_idx, :, :] != 0, axis=(1, 2))

        # Prediction existence (after argmax)
        pred_exists = np.any(y_pred_argmax[:, pair_idx, :, :] != 0, axis=(1, 2))

        TP = np.sum(gt_exists & pred_exists)
        TN = np.sum(~gt_exists & ~pred_exists)
        FP = np.sum(~gt_exists & pred_exists)
        FN = np.sum(gt_exists & ~pred_exists)

        total = len(gt_exists)

        accuracy_exist = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print(f"Total events: {total}")
        print(f"True Positives : {TP}")
        print(f"True Negatives : {TN}")
        print(f"False Positives: {FP}")
        print(f"False Negatives: {FN}")

        print(f"\nExistence Accuracy : {accuracy_exist:.4f}")
        print(f"Precision          : {precision:.4f}")
        print(f"Recall             : {recall:.4f}")
        print(f"Specificity        : {specificity:.4f}")
        print(f"F1 Score           : {f1:.4f}")

        if np.sum(~gt_exists) > 0:
            fp_rate_empty = FP / np.sum(~gt_exists)
            print(f"\nFalse Positive Rate on Empty Pairs: {fp_rate_empty:.4f}")

        # Check for non-zero ground truth to determine valid events
        valid_mask = np.any(y_test[:, pair_idx, :, :] != 0, axis=(1, 2))

        num_valid = np.sum(valid_mask)
        if num_valid == 0:
            print(f"No valid events for pair {pair_idx}, skipping...")
            continue

        print(f"Valid events: {num_valid}/{len(y_test)}")

        # Extract predictions and ground truth for this pair
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
