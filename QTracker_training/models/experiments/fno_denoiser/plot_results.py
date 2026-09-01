"""
Generates team-shareable plots/metrics for the FNO denoiser, without touching
a training job that may still be running:

1. Loss/precision/recall curves over epochs -- pulled from MLflow (already
   logged per-epoch by train.py's MLflowEpochCallback), not from the .out log.
2. A real precision-recall curve (threshold sweep, not just the single
   default-0.5-threshold Precision/Recall train.py tracks) -- computed by
   loading a checkpoint and running it against the validation set directly.
3. A plain-text metrics summary.

Safe to run at any point during training against whatever checkpoint has
been saved so far (train.py's ModelCheckpoint writes *_best.keras on every
val_loss improvement) -- does not require the run to have finished.

Usage (inside the apptainer container, same MLFLOW_TRACKING_URI as training):
    python3 plot_results.py \
        --checkpoint /mnt/data/checkpoints/fno_denoiser/<run>/fno_denoiser_best.keras \
        --val_root_file /mnt/data/data/multi_track/processed_files/mc_events_val.root \
        --mlflow_experiment fno_denoiser \
        --mlflow_run_name <run name, e.g. fno_denoiser_v1_12345678> \
        --output_dir /mnt/data/checkpoints/fno_denoiser/<run>/plots
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from fno_layers import FourierBlock1D  # noqa: E402
from layers import AxialAttention  # noqa: E402
from data_loader import load_data_denoise  # noqa: E402

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def plot_mlflow_curves(experiment_name: str, run_name: str, output_dir: str) -> dict:
    """Pulls per-epoch metric history for the named run and plots loss +
    precision/recall over epochs. Returns the final-epoch values as a dict,
    or {} if the run/experiment can't be found (e.g. MLflow unreachable)."""
    if not MLFLOW_AVAILABLE:
        print("mlflow not available -- skipping epoch-curve plots.")
        return {}

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"MLflow experiment '{experiment_name}' not found -- skipping epoch-curve plots.")
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'" if run_name else "",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        print(f"No run named '{run_name}' found in '{experiment_name}' -- skipping epoch-curve plots.")
        return {}

    run_id = runs.iloc[0]["run_id"]
    client = MlflowClient()

    def history(key):
        hist = client.get_metric_history(run_id, key)
        hist.sort(key=lambda m: m.step)
        return [m.step for m in hist], [m.value for m in hist]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    final_values = {}
    for ax, (train_key, val_key, title) in zip(axes, [
        ("loss", "val_loss", "Loss"),
        ("precision", "val_precision", "Precision (threshold=0.5)"),
        ("recall", "val_recall", "Recall (threshold=0.5)"),
    ]):
        steps_t, vals_t = history(train_key)
        steps_v, vals_v = history(val_key)
        if vals_t:
            ax.plot(steps_t, vals_t, label="train")
            final_values[train_key] = vals_t[-1]
        if vals_v:
            ax.plot(steps_v, vals_v, label="val")
            final_values[val_key] = vals_v[-1]
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle(f"FNO Denoiser -- {run_name}")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "loss_and_metrics_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({len(steps_t)} epochs so far)")
    return final_values


def plot_pr_curve(checkpoint: str, val_root_file: str, max_pairs: int, output_dir: str, sample_events: int) -> dict:
    """Loads the checkpoint, predicts on validation data, and computes a real
    (threshold-swept) pixel-level precision-recall curve."""
    model = tf.keras.models.load_model(
        checkpoint, compile=False,
        custom_objects={"FourierBlock1D": FourierBlock1D, "AxialAttention": AxialAttention},
    )

    X_val, X_clean_val, _yp, _ym = load_data_denoise(val_root_file, multi_track=True, max_pairs=max_pairs)
    if X_val is None:
        raise RuntimeError(f"Could not load validation data from {val_root_file}")

    if sample_events and sample_events < len(X_val):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_val), size=sample_events, replace=False)
        X_val, X_clean_val = X_val[idx], X_clean_val[idx]

    logits = model.predict(X_val, batch_size=64, verbose=1)
    probs = tf.sigmoid(tf.cast(logits, tf.float32)).numpy()

    y_true = X_clean_val.reshape(-1).astype(np.int32)
    y_score = probs.reshape(-1)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"FNO Denoiser -- Precision-Recall ({len(X_val)} val events)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pr_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Precision/recall at a few representative thresholds, for the summary.
    threshold_report = {}
    for t in (0.3, 0.5, 0.7, 0.9):
        pred = (y_score >= t).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        threshold_report[t] = (p, r)

    return {"pr_auc": pr_auc, "n_val_events": len(X_val), "by_threshold": threshold_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_root_file", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--mlflow_experiment", type=str, default="fno_denoiser")
    parser.add_argument("--mlflow_run_name", type=str, default=None, help="Omit to skip MLflow epoch curves.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_events", type=int, default=0, help="Subsample validation events for speed (0 = use all).")
    args = parser.parse_args()

    print(f"\n{'=' * 60}\nMLflow epoch curves\n{'=' * 60}")
    final_values = {}
    if args.mlflow_run_name:
        final_values = plot_mlflow_curves(args.mlflow_experiment, args.mlflow_run_name, args.output_dir)

    print(f"\n{'=' * 60}\nPrecision-recall curve (from checkpoint)\n{'=' * 60}")
    pr_report = plot_pr_curve(args.checkpoint, args.val_root_file, args.max_pairs, args.output_dir, args.sample_events)

    summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FNO Denoiser -- Metrics Summary\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Validation events used for PR curve: {pr_report['n_val_events']}\n\n")
        if final_values:
            f.write("Latest epoch (from MLflow):\n")
            for k, v in final_values.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        f.write(f"PR AUC (pixel-level): {pr_report['pr_auc']:.4f}\n\n")
        f.write("Precision / Recall at fixed thresholds:\n")
        for t, (p, r) in pr_report["by_threshold"].items():
            f.write(f"  threshold={t}: precision={p:.4f}, recall={r:.4f}\n")
    print(f"\nWrote {summary_path}")
    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
