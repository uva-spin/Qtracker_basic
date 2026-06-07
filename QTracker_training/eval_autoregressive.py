"""
Evaluate autoregressive multi-track predictions against ground truth.

Loads the .npz output from AutoregressiveTrackFinder.py and computes:
- Pair-detection confusion matrices (global + per extraction rank)
- Per extraction-rank raw and refined hit accuracy and residual tables
- Residual plots
"""

# ruff: noqa: E402

import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import ROOT  # noqa: F401

import QTracker
from refine import refine_hit_arrays


def plot_residuals(det_ids, res_plus, res_minus, predictions_file, stage_label):
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
    plt.title(f"Per-layer Absolute Residual ({stage_label})")
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(predictions_file))[0]
    fname = f"{base}_{stage_label}_residuals.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()


def plot_confusion_matrix(tp, tn, fp, fn, predictions_file, stage_label):
    """Save a 2x2 pair-existence confusion matrix heatmap."""
    matrix = np.array([[tp, fn], [fp, tn]], dtype=np.int64)
    labels = np.array([["TP", "FN"], ["FP", "TN"]], dtype=object)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted +", "Predicted -"])
    ax.set_yticklabels(["GT +", "GT -"])
    ax.set_title(f"Pair Existence ({stage_label})")

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{matrix[i, j]:,}",
                ha="center",
                va="center",
                color=color,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(predictions_file))[0]
    fname = f"{base}_{stage_label}_confusion_matrix.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()


def print_confusion_matrix(label, tp, tn, fp, fn, predictions_file, stage_label):
    """Print pair-existence confusion matrix and derived metrics."""
    total = tp + tn + fp + fn
    print(f"\n--- {label} Pair Existence Confusion Matrix ---")
    print(f"{'':>20} | {'Predicted +':>11} | {'Predicted -':>11}")
    print(f"{'GT +':>20} | {tp:11,d} | {fn:11,d}")
    print(f"{'GT -':>20} | {fp:11,d} | {tn:11,d}")
    print(f"\nTotal events/slots: {total:,}")

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score   : {f1:.4f}")

    plot_confusion_matrix(tp, tn, fp, fn, predictions_file, stage_label)


def print_global_pair_detection(tp, fn, fp):
    """Print dataset-level pair counts after Hungarian matching."""
    total_gt = tp + fn
    total_pred = tp + fp

    print("\n--- Global Pair Detection (Hungarian matching) ---")
    print(f"Total ground-truth pairs : {total_gt:,}")
    print(f"Total predicted pairs    : {total_pred:,}")
    print(f"Matched pairs (TP)       : {tp:,}")
    print(f"Unmatched GT (FN)        : {fn:,}")
    print(f"Unmatched pred (FP)      : {fp:,}")

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"\nPair detection precision : {precision:.4f}")
    print(f"Pair detection recall    : {recall:.4f}")
    print(f"Pair detection F1        : {f1:.4f}")


def hungarian_match_predictions(y_gt, y_pred, n_gt, n_pred):
    """Match predicted pairs to GT pairs using Hungarian algorithm."""
    if n_gt == 0 or n_pred == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    cost = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            cost[i, j] = np.sum(np.abs(y_pred[i] - y_gt[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    n = min(n_gt, n_pred)
    return col_ind[:n], row_ind[:n]


def compute_true_counts(y_gt, max_pairs, num_events):
    """Count how many non-zero GT pairs each event has."""
    true_counts = np.zeros(num_events, dtype=np.int32)
    for ev in range(num_events):
        for k in range(max_pairs):
            if np.any(y_gt[ev, k, :, :] != 0):
                true_counts[ev] = k + 1
    return true_counts


def print_summary(label, gt_p, gt_m, pred_p, pred_m, mask):
    """Print summary metrics matching evaluate.py output format."""
    res_p = gt_p - pred_p
    res_m = gt_m - pred_m

    acc_p = np.mean(np.abs(res_p[:, mask]) == 0)
    acc_m = np.mean(np.abs(res_m[:, mask]) == 0)
    print(f"\n{label} μ+ accuracy: {acc_p:.4f}")
    print(f"{label} μ- accuracy: {acc_m:.4f}")

    w2_p = np.mean(np.abs(res_p[:, mask]) <= 2)
    w2_m = np.mean(np.abs(res_m[:, mask]) <= 2)
    print(f"\n{label} μ+ within-2 accuracy: {w2_p:.4f}")
    print(f"{label} μ- within-2 accuracy: {w2_m:.4f}")

    m_p = np.mean(np.abs(res_p[:, mask]))
    s_p = np.std(np.abs(res_p[:, mask]))
    m_m = np.mean(np.abs(res_m[:, mask]))
    s_m = np.std(np.abs(res_m[:, mask]))
    print(f"\n--- {label} Absolute Residuals ---")
    print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


def print_per_detector_table(label, res_p, res_m, mask):
    """Print per-detector residual table matching evaluate.py."""
    print(f"\n--- {label} Residuals ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p = np.mean(np.abs(res_p[:, det]))
        s_p = np.std(np.abs(res_p[:, det]))
        m_m = np.mean(np.abs(res_m[:, det]))
        s_m = np.std(np.abs(res_m[:, det]))
        print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


def refine_matched_pairs(pred_p, pred_m, event_indices, detector_ids, element_ids):
    """Refine matched pair predictions using each source event's recorded hits."""
    ref_p = np.zeros_like(pred_p)
    ref_m = np.zeros_like(pred_m)
    by_event: dict[int, list[int]] = {}
    for i, ev in enumerate(event_indices):
        by_event.setdefault(int(ev), []).append(i)

    for ev, indices in by_event.items():
        idx = np.array(indices)
        rp, rm = refine_hit_arrays(
            pred_p[idx],
            pred_m[idx],
            [detector_ids[ev]] * len(idx),
            [element_ids[ev]] * len(idx),
        )
        ref_p[idx] = rp
        ref_m[idx] = rm

    return ref_p, ref_m


def evaluate_hit_quality(
    label,
    gt_p,
    gt_m,
    pred_p,
    pred_m,
    event_indices,
    detector_ids,
    element_ids,
    mask,
    predictions_file,
    plot_label,
):
    """Print and plot raw and refined hit-quality metrics for matched pairs."""
    ref_p, ref_m = refine_matched_pairs(
        pred_p, pred_m, event_indices, detector_ids, element_ids
    )

    raw_res_p = gt_p - pred_p
    raw_res_m = gt_m - pred_m
    ref_res_p = gt_p - ref_p
    ref_res_m = gt_m - ref_m

    dets_used = np.where(mask)[0] + 1

    print_per_detector_table(f"{label} Raw", raw_res_p, raw_res_m, mask)
    plot_residuals(
        dets_used,
        raw_res_p[:, mask],
        raw_res_m[:, mask],
        predictions_file,
        f"{plot_label}_raw",
    )
    print_summary(f"{label} Raw", gt_p, gt_m, pred_p, pred_m, mask)

    print_per_detector_table(f"{label} Refined", ref_res_p, ref_res_m, mask)
    plot_residuals(
        dets_used,
        ref_res_p[:, mask],
        ref_res_m[:, mask],
        predictions_file,
        f"{plot_label}_refined",
    )
    print_summary(f"{label} Refined", gt_p, gt_m, ref_p, ref_m, mask)


def evaluate(args):
    data = np.load(args.predictions_file)

    all_mup = data["all_mup"]  # (max_pairs, num_events, 62)
    all_mum = data["all_mum"]
    n_pairs = data["n_pairs"]  # (num_events,)
    y_muPlus = data["y_muPlus"]  # (num_events, max_pairs, 62) or (num_events, 62)
    y_muMinus = data["y_muMinus"]

    max_pairs = all_mup.shape[0]
    num_events = all_mup.shape[1]

    if len(y_muPlus.shape) == 2:
        y_muPlus = y_muPlus[:, np.newaxis, :]
        y_muMinus = y_muMinus[:, np.newaxis, :]

    if y_muPlus.shape[1] != max_pairs:
        raise ValueError(
            f"GT pair slots ({y_muPlus.shape[1]}) != prediction ranks ({max_pairs})"
        )

    y_gt = np.stack([y_muPlus, y_muMinus], axis=2)  # (num_events, max_pairs, 2, 62)

    detector_ids, element_ids, _, _, _ = QTracker.load_detector_element_data(
        args.root_file
    )
    if len(detector_ids) != num_events:
        raise ValueError(
            f"ROOT events ({len(detector_ids)}) != predictions ({num_events}). "
            "Use the same ROOT file passed to AutoregressiveTrackFinder.py."
        )

    mask = np.ones(62, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    true_counts = compute_true_counts(y_gt, max_pairs, num_events)

    print(f"\n{'=' * 70}")
    print("Autoregressive Multi-Track Evaluation")
    print(f"Events: {num_events}, Max pairs: {max_pairs}")
    print(f"{'=' * 70}")

    gt_exists = np.any(y_gt != 0, axis=(2, 3))
    pred_exists = np.zeros((num_events, max_pairs), dtype=bool)
    for ev in range(num_events):
        for k in range(int(n_pairs[ev])):
            pred_exists[ev, k] = np.any(all_mup[k, ev] != 0) or np.any(
                all_mum[k, ev] != 0
            )

    for k in range(max_pairs):
        gt_k = gt_exists[:, k]
        pred_k = pred_exists[:, k]
        tp = int(np.sum(gt_k & pred_k))
        tn = int(np.sum(~gt_k & ~pred_k))
        fp = int(np.sum(~gt_k & pred_k))
        fn = int(np.sum(gt_k & ~pred_k))

        print(f"\n{'=' * 70}")
        print(f"Rank {k} Pair Existence")
        print(f"{'=' * 70}")
        print_confusion_matrix(
            f"Rank {k}",
            tp,
            tn,
            fp,
            fn,
            args.predictions_file,
            f"rank{k}_existence",
        )

    # Hungarian-match predictions to GT and bucket by extraction rank
    rank_gt_p = [[] for _ in range(max_pairs)]
    rank_gt_m = [[] for _ in range(max_pairs)]
    rank_pred_p = [[] for _ in range(max_pairs)]
    rank_pred_m = [[] for _ in range(max_pairs)]
    rank_event_idx = [[] for _ in range(max_pairs)]
    total_matched_pairs = 0

    for ev in range(num_events):
        n_gt_ev = int(true_counts[ev])
        n_pred_ev = int(n_pairs[ev])

        if n_gt_ev == 0 or n_pred_ev == 0:
            continue

        pred_ev = np.stack(
            [all_mup[:n_pred_ev, ev, :], all_mum[:n_pred_ev, ev, :]],
            axis=1,
        )
        gt_ev = y_gt[ev, :n_gt_ev, :, :]

        gt_order, pred_order = hungarian_match_predictions(
            gt_ev, pred_ev, n_gt_ev, n_pred_ev
        )
        total_matched_pairs += len(gt_order)

        for gi, pi in zip(gt_order, pred_order):
            rank_gt_p[pi].append(gt_ev[gi, 0, :])
            rank_gt_m[pi].append(gt_ev[gi, 1, :])
            rank_pred_p[pi].append(pred_ev[pi, 0, :])
            rank_pred_m[pi].append(pred_ev[pi, 1, :])
            rank_event_idx[pi].append(ev)

    total_gt_pairs = int(np.sum(true_counts))
    total_pred_pairs = int(np.sum(n_pairs))
    global_fn = total_gt_pairs - total_matched_pairs
    global_fp = total_pred_pairs - total_matched_pairs

    print(f"\n{'=' * 70}")
    print("Global Pair Detection")
    print(f"{'=' * 70}")
    print_global_pair_detection(total_matched_pairs, global_fn, global_fp)

    # Per-rank hit-quality evaluation (raw + refined)
    for k in range(max_pairs):
        n_match = len(rank_gt_p[k])
        if n_match == 0:
            continue

        print(f"\n{'=' * 70}")
        print(f"Rank {k} ({n_match} matched pairs)")
        print(f"{'=' * 70}")

        evaluate_hit_quality(
            f"Rank {k}",
            np.array(rank_gt_p[k]),
            np.array(rank_gt_m[k]),
            np.array(rank_pred_p[k]),
            np.array(rank_pred_m[k]),
            rank_event_idx[k],
            detector_ids,
            element_ids,
            mask,
            args.predictions_file,
            f"rank{k}",
        )

    # Combined across all ranks
    all_gt_p = [v for rank in rank_gt_p for v in rank]
    all_gt_m = [v for rank in rank_gt_m for v in rank]
    all_pred_p = [v for rank in rank_pred_p for v in rank]
    all_pred_m = [v for rank in rank_pred_m for v in rank]
    all_event_idx = [v for rank in rank_event_idx for v in rank]

    if all_gt_p:
        print(f"\n{'=' * 70}")
        print(f"All Ranks Combined ({len(all_gt_p)} matched pairs)")
        print(f"{'=' * 70}")

        evaluate_hit_quality(
            "All Ranks",
            np.array(all_gt_p),
            np.array(all_gt_m),
            np.array(all_pred_p),
            np.array(all_pred_m),
            all_event_idx,
            detector_ids,
            element_ids,
            mask,
            args.predictions_file,
            "all_ranks",
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate autoregressive multi-track predictions."
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        help="Path to .npz file from AutoregressiveTrackFinder.py",
    )
    parser.add_argument(
        "root_file",
        type=str,
        help="Path to the ROOT file used to generate predictions (for hit refinement).",
    )
    args = parser.parse_args()
    evaluate(args)
