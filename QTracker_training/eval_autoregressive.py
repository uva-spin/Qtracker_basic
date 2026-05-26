"""
Evaluate autoregressive multi-track predictions against ground truth.

Loads the .npz output from AutoregressiveTrackFinder.py and computes:
- Pair existence metrics (TP/TN/FP/FN)
- Per-detector residuals for each extracted pair
- Per-pair accuracy and chi-squared
"""

# ruff: noqa: E402

import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def chi_squared(y_true, y_pred):
    residuals = y_true - y_pred
    sigma = np.std(y_true, axis=0) + 1e-6
    res_norm = residuals / sigma
    chi2 = np.sum(res_norm**2, axis=1)
    return np.mean(chi2)


def hungarian_match_predictions(y_gt, y_pred, n_gt, n_pred):
    """
    Match predicted pairs to GT pairs using Hungarian algorithm.

    The autoregressive approach produces pairs in confidence order, not canonical order.
    This function finds the optimal assignment to minimize total element-ID residuals.

    Args:
        y_gt: (max_gt_pairs, 2, 62) GT pairs for one event
        y_pred: (max_pred_iters, 2, 62) predicted pairs for one event
        n_gt: number of active GT pairs
        n_pred: number of active predicted pairs

    Returns:
        gt_order: (min(n_gt, n_pred),) indices into y_gt
        pred_order: (min(n_gt, n_pred),) indices into y_pred
    """
    if n_gt == 0 or n_pred == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n = min(n_gt, n_pred)

    cost = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            cost[i, j] = np.sum(np.abs(y_pred[i] - y_gt[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind[:n], row_ind[:n]


def plot_residuals(det_ids, res_plus, res_minus, output_dir, label):
    mean_p = np.nanmean(np.abs(res_plus), axis=0)
    std_p = np.nanstd(np.abs(res_plus), axis=0)
    mean_m = np.nanmean(np.abs(res_minus), axis=0)
    std_m = np.nanstd(np.abs(res_minus), axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(det_ids, mean_p, yerr=std_p, marker="o", label="μ+ mean±σ")
    plt.errorbar(det_ids, mean_m, yerr=std_m, marker="s", label="μ- mean±σ")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Detector Layer")
    plt.ylabel("Absolute Residual")
    plt.title(f"Per-layer Absolute Residual ({label})")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"autoregressive_{label}_residuals.png"))
    plt.close()


def evaluate(args):
    data = np.load(args.predictions_file)

    all_mup = data["all_mup"]  # (max_iters, num_events, 62)
    all_mum = data["all_mum"]
    all_conf = data["all_conf"]  # (max_iters, num_events)
    n_pairs = data["n_pairs"]  # (num_events,)
    y_muPlus = data["y_muPlus"]  # (num_events, max_pairs, 62) or (num_events, 62)
    y_muMinus = data["y_muMinus"]

    max_iters = all_mup.shape[0]
    num_events = all_mup.shape[1]

    # Handle both single-track and multi-track GT formats
    if len(y_muPlus.shape) == 2:
        y_muPlus = y_muPlus[:, np.newaxis, :]
        y_muMinus = y_muMinus[:, np.newaxis, :]

    gt_max_pairs = y_muPlus.shape[1]

    # Stack GT: (num_events, max_pairs, 2, 62)
    y_gt = np.stack([y_muPlus, y_muMinus], axis=2)

    mask = np.ones(62, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    print(f"\n{'=' * 70}")
    print("Autoregressive Multi-Track Evaluation")
    print(f"Events: {num_events}, Max iters: {max_iters}, GT max pairs: {gt_max_pairs}")
    print(f"{'=' * 70}")

    # Compute true pair counts
    true_counts = np.zeros(num_events, dtype=np.int32)
    for ev in range(num_events):
        for k in range(gt_max_pairs):
            if np.any(y_gt[ev, k, :, :] != 0, axis=(0, 1)):
                true_counts[ev] = k + 1

    # Count distribution
    print("\n--- Pair Count Distribution ---")
    print("  k  | Predicted | GT")
    for k in range(max(max_iters, gt_max_pairs) + 1):
        pred_count = np.sum(n_pairs == k)
        gt_count = np.sum(true_counts == k)
        print(f"  {k}: {pred_count:6d}   | {gt_count:6d}")

    # --- Hungarian-matched evaluation ---
    # The autoregressive approach finds pairs in confidence order, NOT canonical order.
    # We must match predicted pairs to GT pairs optimally before computing residuals.
    print(f"\n{'=' * 70}")
    print("Hungarian-Matched Evaluation")
    print(f"{'=' * 70}")

    # Pair count accuracy
    count_accuracy = np.mean(n_pairs == true_counts)
    print(f"Pair count accuracy: {count_accuracy:.4f}")
    print(f"  Overcount:  {np.sum(n_pairs > true_counts)}")
    print(f"  Undercount: {np.sum(n_pairs < true_counts)}")
    print(f"  Correct:    {np.sum(n_pairs == true_counts)}")

    # Collect matched residuals and GT/pred pairs for chi-squared
    all_matched_p_res = []
    all_matched_m_res = []
    matched_gt_refs = []  # (gt_ev, gi) tuples for chi-squared
    matched_pred_refs = []  # (pred_ev, pi) tuples for chi-squared
    total_matched_pairs = 0

    for ev in range(num_events):
        n_gt_ev = int(true_counts[ev])
        n_pred_ev = int(n_pairs[ev])

        if n_gt_ev == 0 or n_pred_ev == 0:
            continue

        # Build pred array for this event: (n_pred, 2, 62)
        pred_ev = np.stack(
            [
                all_mup[:n_pred_ev, ev, :],
                all_mum[:n_pred_ev, ev, :],
            ],
            axis=1,
        )  # (n_pred, 2, 62)

        gt_ev = y_gt[ev, :n_gt_ev, :, :]  # (n_gt, 2, 62)

        gt_order, pred_order = hungarian_match_predictions(
            gt_ev, pred_ev, n_gt_ev, n_pred_ev
        )

        for gi, pi in zip(gt_order, pred_order):
            all_matched_p_res.append(gt_ev[gi, 0, :] - pred_ev[pi, 0, :])
            all_matched_m_res.append(gt_ev[gi, 1, :] - pred_ev[pi, 1, :])
            matched_gt_refs.append((gt_ev, gi))
            matched_pred_refs.append((pred_ev, pi))
            total_matched_pairs += 1

    if total_matched_pairs > 0:
        matched_p_res = np.array(all_matched_p_res)
        matched_m_res = np.array(all_matched_m_res)

        print(f"\nTotal matched pairs: {total_matched_pairs}")

        acc_p = np.mean(np.abs(matched_p_res[:, mask]) == 0)
        acc_m = np.mean(np.abs(matched_m_res[:, mask]) == 0)
        print(f"Matched μ+ accuracy: {acc_p:.4f}")
        print(f"Matched μ- accuracy: {acc_m:.4f}")

        acc_p_2 = np.mean(np.abs(matched_p_res[:, mask]) <= 2)
        acc_m_2 = np.mean(np.abs(matched_m_res[:, mask]) <= 2)
        print(f"Matched μ+ within-2: {acc_p_2:.4f}")
        print(f"Matched μ- within-2: {acc_m_2:.4f}")

        print(
            f"Matched μ+ mean |residual|: {np.mean(np.abs(matched_p_res[:, mask])):.3f}"
        )
        print(
            f"Matched μ- mean |residual|: {np.mean(np.abs(matched_m_res[:, mask])):.3f}"
        )

        # Chi-squared: pass matched GT and matched predictions (not residuals)
        all_matched_p_gt = np.array([gt_ev[gi, 0, :] for gt_ev, gi in matched_gt_refs])
        all_matched_p_pred = np.array(
            [pred_ev[pi, 0, :] for pred_ev, pi in matched_pred_refs]
        )
        all_matched_m_gt = np.array([gt_ev[gi, 1, :] for gt_ev, gi in matched_gt_refs])
        all_matched_m_pred = np.array(
            [pred_ev[pi, 1, :] for pred_ev, pi in matched_pred_refs]
        )
        chi2_p = chi_squared(all_matched_p_gt, all_matched_p_pred)
        chi2_m = chi_squared(all_matched_m_gt, all_matched_m_pred)
        print(f"Matched μ+ Chi²: {chi2_p:.3f}")
        print(f"Matched μ- Chi²: {chi2_m:.3f}")

        # Plot matched residuals
        dets_used = np.where(mask)[0] + 1
        plot_residuals(
            dets_used,
            matched_p_res[:, mask],
            matched_m_res[:, mask],
            os.path.join("plots", "autoregressive"),
            "matched",
        )
    else:
        print("No matched pairs found.")

    # --- Per-iteration unmatched evaluation (supplementary) ---
    # Also report per-iteration stats for diagnostic purposes
    for it in range(min(max_iters, gt_max_pairs)):
        print(f"\n--- Iteration {it} (unmatched, supplementary) ---")

        gt_exists = np.any(y_gt[:, it, :, :] != 0, axis=(1, 2))
        pred_exists = n_pairs > it

        TP = np.sum(gt_exists & pred_exists)
        FN = np.sum(gt_exists & ~pred_exists)
        FP = np.sum(~gt_exists & pred_exists)

        valid_mask = gt_exists & pred_exists
        if np.sum(valid_mask) == 0:
            print(f"  No valid events for iteration {it}.")
            continue

        conf = all_conf[it, valid_mask]
        print(f"  Active: {np.sum(pred_exists)}, TP: {TP}, FN: {FN}, FP: {FP}")
        print(f"  Mean confidence: {np.mean(conf):.4f} ± {np.std(conf):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate autoregressive multi-track predictions."
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to .npz file from AutoregressiveTrackFinder.py",
    )
    args = parser.parse_args()
    evaluate(args)
