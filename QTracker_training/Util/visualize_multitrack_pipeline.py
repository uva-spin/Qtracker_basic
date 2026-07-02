"""
Visualize the MultiTrackFinder pipeline for a single event:
    1. Noisy input hit matrix
    2. Denoised output (model's shared denoiser head)
    3. Final predicted tracks (per pair, Hungarian-matched to ground truth) overlaid
       on the denoised matrix, with ground truth shown for comparison.

Usage:
    python Util/visualize_multitrack_pipeline.py <root_file> <model_path> --event 0
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.layers import AxialAttention
from models.data_loader import load_data_denoise

PAIR_COLORS = ["red", "dodgerblue", "limegreen"]


def load_single_event(root_file: str, event_idx: int, max_pairs: int):
    X, X_clean, y_mup, y_mum = load_data_denoise(
        root_file, multi_track=True, max_pairs=max_pairs
    )
    if X is None:
        return None, None, None, None
    return (
        X[event_idx : event_idx + 1],
        X_clean[event_idx : event_idx + 1],
        y_mup[event_idx],
        y_mum[event_idx],
    )


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"AxialAttention": AxialAttention},
    )


def run_inference(model: tf.keras.Model, X_input: np.ndarray):
    """Returns (denoised (62, 201), pred_argmax (max_pairs, 2, 62))."""
    denoise_out, segment_out = model.predict(tf.cast(X_input, tf.float32), verbose=0)
    pred_argmax = np.argmax(segment_out, axis=-1)[0]  # (max_pairs, 2, 62)
    return denoise_out[0, :, :, 0], pred_argmax


def match_pairs_to_truth(
    pred_argmax: np.ndarray, y_mup_true: np.ndarray, y_mum_true: np.ndarray
) -> np.ndarray:
    """Hungarian-match predicted pair slots to ground-truth pair ordering."""
    y_true = np.stack([y_mup_true, y_mum_true], axis=1)  # (max_pairs, 2, 62)
    n = pred_argmax.shape[0]
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.mean(
                np.abs(pred_argmax[i].astype(float) - y_true[j].astype(float))
            )
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(pred_argmax)
    for k in range(len(row_ind)):
        aligned[col_ind[k]] = pred_argmax[row_ind[k]]
    return aligned


def plot_pipeline(
    X_input: np.ndarray,
    denoised: np.ndarray,
    pred_aligned: np.ndarray,
    y_mup_true: np.ndarray,
    y_mum_true: np.ndarray,
    output_path: str,
    event_idx: int,
):
    cmap_hits = LinearSegmentedColormap.from_list("hits", ["white", "navy"], N=256)
    max_pairs = pred_aligned.shape[0]
    detectors = np.arange(62)

    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: noisy input
    axs[0].imshow(
        X_input.T, aspect="auto", origin="lower", cmap=cmap_hits, vmin=0, vmax=1,
        extent=[0, 62, 0, 201],
    )
    axs[0].set_title(f"1. Noisy Input — Event {event_idx}", fontsize=13, weight="bold")
    axs[0].set_xlabel("Detector ID")
    axs[0].set_ylabel("Element ID")
    n_hits = int(np.sum(X_input > 0))
    axs[0].text(
        0.02, 0.98, f"Total hits: {n_hits}", transform=axs[0].transAxes,
        va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Panel 2: denoised output
    im1 = axs[1].imshow(
        denoised.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
        extent=[0, 62, 0, 201],
    )
    axs[1].set_title("2. Denoised Output", fontsize=13, weight="bold")
    axs[1].set_xlabel("Detector ID")
    axs[1].set_ylabel("Element ID")
    fig.colorbar(im1, ax=axs[1], label="Hit Probability", fraction=0.046)

    # Panel 3: final predicted tracks vs ground truth, on top of denoised background
    axs[2].imshow(
        denoised.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1,
        extent=[0, 62, 0, 201], alpha=0.6,
    )
    residual_lines = []
    for p in range(max_pairs):
        color = PAIR_COLORS[p % len(PAIR_COLORS)]
        gt_plus, gt_minus = y_mup_true[p], y_mum_true[p]
        pr_plus, pr_minus = pred_aligned[p, 0], pred_aligned[p, 1]

        has_truth = np.any(gt_plus > 0) or np.any(gt_minus > 0)
        if not has_truth:
            continue

        gt_plus_nz = gt_plus > 0
        gt_minus_nz = gt_minus > 0
        pr_plus_nz = pr_plus > 0
        pr_minus_nz = pr_minus > 0

        axs[2].scatter(
            detectors[gt_plus_nz], gt_plus[gt_plus_nz], facecolors="none",
            edgecolors=color, s=70, marker="o", linewidths=1.8,
            label=f"Pair {p} μ+ truth",
        )
        axs[2].scatter(
            detectors[gt_minus_nz], gt_minus[gt_minus_nz], facecolors="none",
            edgecolors=color, s=70, marker="s", linewidths=1.8,
            label=f"Pair {p} μ− truth",
        )
        axs[2].scatter(
            detectors[pr_plus_nz], pr_plus[pr_plus_nz], c=color, s=25, marker="o",
            label=f"Pair {p} μ+ pred",
        )
        axs[2].scatter(
            detectors[pr_minus_nz], pr_minus[pr_minus_nz], c=color, s=25, marker="s",
            label=f"Pair {p} μ− pred",
        )

        res_plus = np.mean(np.abs(pr_plus[gt_plus_nz] - gt_plus[gt_plus_nz])) if gt_plus_nz.any() else np.nan
        res_minus = np.mean(np.abs(pr_minus[gt_minus_nz] - gt_minus[gt_minus_nz])) if gt_minus_nz.any() else np.nan
        residual_lines.append(f"Pair {p}: μ+ resid={res_plus:.2f}, μ− resid={res_minus:.2f}")

    axs[2].set_title("3. Predicted Tracks vs Truth", fontsize=13, weight="bold")
    axs[2].set_xlabel("Detector ID")
    axs[2].set_ylabel("Element ID")
    axs[2].set_xlim(0, 62)
    axs[2].set_ylim(0, 201)
    axs[2].legend(fontsize=7, loc="upper right", ncol=2)
    if residual_lines:
        axs[2].text(
            0.02, 0.02, "\n".join(residual_lines), transform=axs[2].transAxes,
            va="bottom", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

    fig.suptitle("MultiTrackFinder Pipeline", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MultiTrackFinder pipeline for one event")
    parser.add_argument("root_file", type=str, help="Validation ROOT file")
    parser.add_argument("model_path", type=str, help="Path to trained multi_track_finder.keras checkpoint")
    parser.add_argument("--event", type=int, default=0, help="Event index to visualize")
    parser.add_argument("--max_pairs", type=int, default=3, help="Number of pair slots the model was trained with")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)

    print(f"Loading event {args.event} from {args.root_file}...")
    X_event, X_clean, y_mup_true, y_mum_true = load_single_event(
        args.root_file, args.event, args.max_pairs
    )
    if X_event is None:
        print("Error loading data!")
        return

    print("Running inference...")
    denoised, pred_argmax = run_inference(model, X_event)
    pred_aligned = match_pairs_to_truth(pred_argmax, y_mup_true, y_mum_true)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model_path))[0]
        output_file = f"multitrack_pipeline_event{args.event}_{base}.png"
    else:
        output_file = args.output

    plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, output_file)

    plot_pipeline(
        X_event[0, :, :, 0], denoised, pred_aligned, y_mup_true, y_mum_true,
        output_path, args.event,
    )


if __name__ == "__main__":
    main()
