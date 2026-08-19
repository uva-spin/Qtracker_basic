# ruff: noqa: E402
"""
Evaluate an ensemble of independent single-pair track finders.

Loads N sub-models from a checkpoint directory, runs each on the validation set,
stacks their predictions into (B, N, 2, 62), then applies Hungarian matching
to assign model outputs to ground-truth pair slots.  Reports the same per-pair
accuracy/residual metrics as eval_multi_track.py for direct comparison.
"""

import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import importlib.util
import ROOT  # noqa: F401

from models import data_loader

# Load layers.py by absolute path — ROOT import above can corrupt sys.modules['models']
_layers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "layers.py")
_spec = importlib.util.spec_from_file_location("models.layers", _layers_path)
_layers_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layers_mod)
AxialAttention = _layers_mod.AxialAttention


def load_sub_models(model_dir: str, n_models: int, suffix: str = "_best") -> list:
    """Load ensemble_model_{i}{suffix}.keras from model_dir."""
    models = []
    custom = {"AxialAttention": AxialAttention}
    for i in range(n_models):
        path = os.path.join(model_dir, f"ensemble_model_{i}{suffix}.keras")
        if not os.path.exists(path):
            # fall back to final checkpoint
            path = os.path.join(model_dir, f"ensemble_model_{i}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sub-model {i} not found in {model_dir}")
        m = tf.keras.models.load_model(path, compile=False, custom_objects=custom)
        models.append(m)
        print(f"Loaded sub-model {i}: {path}")
    return models


def predict_ensemble(models: list, X: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    """
    Run all sub-models on X in chunks and stack predictions.

    Returns:
        y_pred_argmax: (N_events, n_models, 2, 62) — element-ID predictions
    """
    n_models = len(models)
    n_events = X.shape[0]
    all_preds = [[] for _ in range(n_models)]

    for start in range(0, n_events, chunk_size):
        X_chunk = tf.cast(X[start:start + chunk_size], tf.float32)
        for i, m in enumerate(models):
            _, seg_out = m.predict(X_chunk, verbose=0)  # seg_out: (B,1,2,62,201)
            all_preds[i].append(np.argmax(seg_out[:, 0], axis=-1).astype(np.int32))  # (B,2,62)

    # Stack: (n_events, n_models, 2, 62)
    stacked = np.stack([np.concatenate(all_preds[i], axis=0) for i in range(n_models)], axis=1)
    return stacked


def find_best_permutation(y_pred_argmax: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Hungarian-match predicted model slots to ground-truth pair ordering."""
    num_events, pred_pairs, _, _ = y_pred_argmax.shape
    gt_pairs = y_test.shape[1]
    n = min(pred_pairs, gt_pairs)
    y_pred_reassigned = np.zeros_like(y_pred_argmax)
    for ev in range(num_events):
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = np.mean(np.abs(
                    y_pred_argmax[ev, i].astype(float) - y_test[ev, j].astype(float)
                ))
        row_ind, col_ind = linear_sum_assignment(cost)
        for k in range(len(row_ind)):
            y_pred_reassigned[ev, col_ind[k]] = y_pred_argmax[ev, row_ind[k]]
    return y_pred_reassigned


def evaluate(args: argparse.Namespace) -> None:
    # Load data
    print("Loading validation data...")
    X_test, _, y_mup_test, y_mum_test = data_loader.load_data_denoise(
        args.val_root_file, multi_track=True, max_pairs=args.n_models
    )
    if X_test is None:
        print("Error loading data.")
        return
    y_test = np.stack([y_mup_test, y_mum_test], axis=2)  # (N, n_models, 2, 62)
    print(f"Loaded {X_test.shape[0]} validation events.")

    # Load models
    print(f"Loading {args.n_models} sub-models from {args.model_dir}...")
    models = load_sub_models(args.model_dir, args.n_models, suffix="_best")

    # Predict
    print("Running ensemble inference...")
    y_pred_argmax = predict_ensemble(models, X_test)  # (N, n_models, 2, 62)

    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION — PER-MODEL PREDICTIONS (raw slot order)")
    print("=" * 70)
    _report_per_pair(y_pred_argmax, y_test, label="Raw (no permutation alignment)")

    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION — HUNGARIAN-MATCHED")
    print("=" * 70)
    y_pred_best = find_best_permutation(y_pred_argmax, y_test)
    _report_per_pair(y_pred_best, y_test, label="Hungarian-matched")

    # Per-pair summary
    print("\n" + "=" * 70)
    print("SUMMARY (Hungarian-matched, non-zero GT slots only)")
    print("=" * 70)
    for pair_idx in range(args.n_models):
        gt_p = y_test[:, pair_idx, 0, :]
        gt_m = y_test[:, pair_idx, 1, :]
        pr_p = y_pred_best[:, pair_idx, 0, :]
        pr_m = y_pred_best[:, pair_idx, 1, :]
        valid = np.any(gt_p != 0, axis=1) | np.any(gt_m != 0, axis=1)
        if not np.any(valid):
            continue
        gp, gm = gt_p[valid], gt_m[valid]
        pp, pm = pr_p[valid], pr_m[valid]
        nz_p = gp != 0
        nz_m = gm != 0
        acc_p = np.mean((pp == gp)[nz_p]) if nz_p.any() else float("nan")
        acc_m = np.mean((pm == gm)[nz_m]) if nz_m.any() else float("nan")
        res_p = np.mean(np.abs((pp - gp)[nz_p])) if nz_p.any() else float("nan")
        res_m = np.mean(np.abs((pm - gm)[nz_m])) if nz_m.any() else float("nan")
        print(f"  Model/Pair {pair_idx}: μ+ acc={acc_p:.4f}, μ- acc={acc_m:.4f} | "
              f"μ+ residual={res_p:.3f} ch, μ- residual={res_m:.3f} ch  (N={np.sum(valid)})")

    # Overall line — matches eval_multi_track.py format for direct comparison
    gp_all = y_test[:, :, 0, :].reshape(-1, 62)
    gm_all = y_test[:, :, 1, :].reshape(-1, 62)
    pp_all = y_pred_best[:, :, 0, :].reshape(-1, 62)
    pm_all = y_pred_best[:, :, 1, :].reshape(-1, 62)
    nz_p = gp_all != 0
    nz_m = gm_all != 0
    oacc_p = np.mean((pp_all == gp_all)[nz_p]) if nz_p.any() else float("nan")
    oacc_m = np.mean((pm_all == gm_all)[nz_m]) if nz_m.any() else float("nan")
    ores_p = np.mean(np.abs((pp_all - gp_all)[nz_p])) if nz_p.any() else float("nan")
    ores_m = np.mean(np.abs((pm_all - gm_all)[nz_m])) if nz_m.any() else float("nan")
    print(f"\nOverall best-permutation | acc μ+/μ-: {oacc_p:.4f}/{oacc_m:.4f} | "
          f"mean residual μ+/μ-: {ores_p:.1f}/{ores_m:.1f} ch")


def _report_per_pair(y_pred: np.ndarray, y_test: np.ndarray, label: str) -> None:
    n_pairs = y_pred.shape[1]
    print(f"\n  [{label}]")
    for pair_idx in range(n_pairs):
        gt_p = y_test[:, pair_idx, 0, :]
        gt_m = y_test[:, pair_idx, 1, :]
        pr_p = y_pred[:, pair_idx, 0, :]
        pr_m = y_pred[:, pair_idx, 1, :]
        valid = np.any(gt_p != 0, axis=1) | np.any(gt_m != 0, axis=1)
        if not np.any(valid):
            continue
        gp, gm = gt_p[valid], gt_m[valid]
        pp, pm = pr_p[valid], pr_m[valid]
        nz_p = gp != 0
        nz_m = gm != 0
        acc_p = np.mean((pp == gp)[nz_p]) if nz_p.any() else float("nan")
        acc_m = np.mean((pm == gm)[nz_m]) if nz_m.any() else float("nan")
        res_p = np.mean(np.abs((pp - gp)[nz_p])) if nz_p.any() else float("nan")
        res_m = np.mean(np.abs((pm - gm)[nz_m])) if nz_m.any() else float("nan")
        print(f"    Pair {pair_idx}: μ+ acc={acc_p:.4f} res={res_p:.2f}ch | "
              f"μ- acc={acc_m:.4f} res={res_m:.2f}ch  (N={np.sum(valid)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate independent-model ensemble for multi-track finding.")
    parser.add_argument("val_root_file", type=str, help="Validation ROOT file.")
    parser.add_argument("model_dir", type=str, help="Directory containing ensemble_model_{i}_best.keras files.")
    parser.add_argument("--n_models", type=int, default=3, help="Number of sub-models in the ensemble.")
    args = parser.parse_args()
    evaluate(args)
