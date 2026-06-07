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

from models import data_loader
from models.layers import AxialAttention
import QTracker
from refine import refine_hit_arrays


def plot_residuals(det_ids, res_plus, res_minus, model_path, stage_label, pair_idx):
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
    plt.title(
        f"Pair {pair_idx + 1} Per-layer Absolute Residual ({stage_label.capitalize()})"
    )
    plt.legend()
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(model_path))[0]
    fname = f"{base}_pair{pair_idx + 1}_{stage_label}_residuals.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots", "multi_track")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()


def min_perm_match(y_pred_argmax, y_true, n_pairs):
    """
    For each event, reorder predicted pair slots to minimize total L1 residual vs GT.

    Args:
        y_pred_argmax: (B, N, 2, 62) int32 predicted element IDs
        y_true: (B, N, 2, 62) int32 GT element IDs
        n_pairs: int, N

    Returns:
        matched_pred: (B, N, 2, 62) int32 -- predictions reordered to best match GT slot order
    """
    perms = list(itertools.permutations(range(n_pairs)))
    B = y_pred_argmax.shape[0]
    matched_pred = np.zeros_like(y_pred_argmax)

    for b in range(B):
        best_cost = np.inf
        best_perm = list(range(n_pairs))
        for perm in perms:
            # cost: sum over i of L1(pred[i], true[perm[i]])
            cost = sum(
                np.sum(np.abs(y_pred_argmax[b, i] - y_true[b, perm[i]]))
                for i in range(n_pairs)
            )
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        # matched_pred[b, best_perm[i], :, :] = y_pred_argmax[b, i, :, :]
        # (pred slot i is matched to GT slot best_perm[i])
        for i, j in enumerate(best_perm):
            matched_pred[b, j] = y_pred_argmax[b, i]

    return matched_pred


def evaluate_model(args):
    n_pairs = args.n_pairs

    # 1. Load data (always multi_track=True)
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(
        args.root_file, multi_track=True, max_pairs=n_pairs
    )
    if X_test is None:
        return
    # y_muPlus_test: (B, N, 62), y_muMinus_test: (B, N, 62)
    # Stack to (B, N, 2, 62):
    y_true = np.stack([y_muPlus_test, y_muMinus_test], axis=2)

    # 2. Load detector data for refinement
    det_test, elem_test, _, _, _ = QTracker.load_detector_element_data(args.root_file)

    # 3. Detector mask (same as evaluate.py)
    mask = np.ones(62, dtype=bool)
    mask[6:12] = False
    mask[54:62] = False

    # 4. Load model
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects={"AxialAttention": AxialAttention},
    )

    # 5. Predict in chunks (segment output = index 1, always 5D (B,N,2,62,201))
    preds = []
    chunk_size = 128
    for i in range(0, len(X_test), chunk_size):
        X_chunk = tf.cast(X_test[i : i + chunk_size], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        preds.append(y_chunk[1])  # segment output
    y_pred = np.concatenate(preds, axis=0)  # (B, N, 2, 62, 201)

    # 6. argmax over element IDs
    y_pred_argmax = np.argmax(y_pred, axis=-1).astype(np.int32)  # (B, N, 2, 62)

    # 7. Permutation matching
    matched_pred = min_perm_match(y_pred_argmax, y_true, n_pairs)  # (B, N, 2, 62)

    # 8. Per-pair metrics
    for pair_idx in range(n_pairs):
        print(f"\n{'=' * 60}")
        print(f"  Pair {pair_idx + 1} of {n_pairs}")
        print(f"{'=' * 60}")

        y_p_true = y_true[:, pair_idx, 0, :].astype(np.int32)  # (B, 62)
        y_m_true = y_true[:, pair_idx, 1, :].astype(np.int32)  # (B, 62)
        y_p_raw = matched_pred[:, pair_idx, 0, :]  # (B, 62)
        y_m_raw = matched_pred[:, pair_idx, 1, :]  # (B, 62)

        raw_p_res = y_p_true - y_p_raw
        raw_m_res = y_m_true - y_m_raw

        # Raw residuals per detector (same as evaluate.py)
        print("\n--- Raw Residuals (Before Refinement) ---")
        print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        for det in np.where(mask)[0]:
            m_p = np.mean(np.abs(raw_p_res[:, det]))
            s_p = np.std(np.abs(raw_p_res[:, det]))
            m_m = np.mean(np.abs(raw_m_res[:, det]))
            s_m = np.std(np.abs(raw_m_res[:, det]))
            print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

        dets_used = np.where(mask)[0] + 1
        plot_residuals(
            dets_used,
            raw_p_res[:, mask],
            raw_m_res[:, mask],
            args.model_path,
            "raw",
            pair_idx,
        )

        # Refine
        ref_p, ref_m = refine_hit_arrays(y_p_raw, y_m_raw, det_test, elem_test)
        ref_p_res = y_p_true - ref_p
        ref_m_res = y_m_true - ref_m

        # Refined residuals per detector
        print("\n--- Refined Residuals (After Refinement) ---")
        print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
        for det in np.where(mask)[0]:
            m_p = np.mean(np.abs(ref_p_res[:, det]))
            s_p = np.std(np.abs(ref_p_res[:, det]))
            m_m = np.mean(np.abs(ref_m_res[:, det]))
            s_m = np.std(np.abs(ref_m_res[:, det]))
            print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

        plot_residuals(
            dets_used,
            ref_p_res[:, mask],
            ref_m_res[:, mask],
            args.model_path,
            "refined",
            pair_idx,
        )

        # Accuracy metrics (NO chi-squared)
        acc_p = np.mean(np.abs(raw_p_res) == 0)
        acc_m = np.mean(np.abs(raw_m_res) == 0)
        print(f"\nRaw μ+ exact-match accuracy: {acc_p:.4f}")
        print(f"Raw μ- exact-match accuracy: {acc_m:.4f}")

        acc_p = np.mean(np.abs(raw_p_res) <= 2)
        acc_m = np.mean(np.abs(raw_m_res) <= 2)
        print(f"Raw μ+ within-2 accuracy: {acc_p:.4f}")
        print(f"Raw μ- within-2 accuracy: {acc_m:.4f}")

        acc_p = np.mean(np.abs(ref_p_res) == 0)
        acc_m = np.mean(np.abs(ref_m_res) == 0)
        print(f"\nRefined μ+ exact-match accuracy: {acc_p:.4f}")
        print(f"Refined μ- exact-match accuracy: {acc_m:.4f}")

        acc_p = np.mean(np.abs(ref_p_res) <= 2)
        acc_m = np.mean(np.abs(ref_m_res) <= 2)
        print(f"Refined μ+ within-2 accuracy: {acc_p:.4f}")
        print(f"Refined μ- within-2 accuracy: {acc_m:.4f}")

        # Global absolute residuals
        print("\n--- Raw Global Absolute Residuals ---")
        m_p, s_p = np.mean(np.abs(raw_p_res)), np.std(np.abs(raw_p_res))
        m_m, s_m = np.mean(np.abs(raw_m_res)), np.std(np.abs(raw_m_res))
        print(f"μ+ mean={m_p:.3f} std={s_p:.3f} | μ- mean={m_m:.3f} std={s_m:.3f}")

        print("\n--- Refined Global Absolute Residuals ---")
        m_p, s_p = np.mean(np.abs(ref_p_res)), np.std(np.abs(ref_p_res))
        m_m, s_m = np.mean(np.abs(ref_m_res)), np.std(np.abs(ref_m_res))
        print(f"μ+ mean={m_p:.3f} std={s_p:.3f} | μ- mean={m_m:.3f} std={s_m:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained fixed-N multi-track finder models."
    )
    parser.add_argument("root_file", type=str, help="Path to the val/test ROOT file.")
    parser.add_argument(
        "model_path", type=str, help="Path to the saved model (.keras or .h5)."
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=1,
        help="Fixed number of dimuon pairs the model was trained with (1-3).",
    )
    args = parser.parse_args()

    print(f"\nEvaluating {args.model_path} (n_pairs={args.n_pairs})...")
    evaluate_model(args)
