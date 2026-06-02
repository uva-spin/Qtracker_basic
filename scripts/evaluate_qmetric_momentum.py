#!/usr/bin/env python3
"""
Evaluate downstream momentum reconstruction after applying QMetric track-selection thresholds.

Input is a QMetric pattern table produced with:
    --save_candidate_arrays 1 --save_truth_momentum 1

Each CSV row is one candidate track. This script selects rows with
qmetric_score >= threshold, reconstructs momentum for those selected tracks,
and summarizes threshold-level momentum error. It does not change QTracker.py.
"""

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

NUM_DETECTORS = 62
INACTIVE_SLICES = (
    (7, 12),
    (55, 58),
    (59, 62),
)


def make_active_mask():
    mask = np.ones(NUM_DETECTORS, dtype=bool)
    for start, stop in INACTIVE_SLICES:
        mask[start:stop] = False
    return mask


ACTIVE_MASK = make_active_mask()


def parse_thresholds(text):
    if text is None or str(text).strip() == "":
        return [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    return [float(item) for item in str(text).split(",") if item.strip()]


def check_required_columns(df, score_col):
    missing = []
    for col in [score_col, "charge", "true_px", "true_py", "true_pz"]:
        if col not in df.columns:
            missing.append(col)
    for det in range(NUM_DETECTORS):
        key = "%02d" % det
        for prefix in ["candidate_elem_", "candidate_drift_"]:
            col = prefix + key
            if col not in df.columns:
                missing.append(col)
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(
            "Input CSV is missing required columns. Rebuild the pattern table with "
            "--save_candidate_arrays 1 --save_truth_momentum 1. Missing: " + preview
        )


def print_environment():
    print("[env] python:", sys.executable)
    try:
        import tensorflow as tf
        print("[env] tensorflow:", tf.__version__, tf.__file__)
        print("[env] physical GPUs:", tf.config.list_physical_devices("GPU"))
    except Exception as exc:
        print("[env] tensorflow import failed:", exc)
    try:
        import keras
        print("[env] keras:", keras.__version__, keras.__file__)
    except Exception as exc:
        print("[env] keras import failed:", exc)


def load_momentum_model(path):
    import tensorflow as tf
    from tensorflow.keras.losses import MeanSquaredError

    print("[load] momentum model:", path)
    custom_objects = {
        "mse": MeanSquaredError(),
        "mean_squared_error": MeanSquaredError(),
    }
    return tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)


def build_momentum_input(df):
    elem_cols = ["candidate_elem_%02d" % det for det in range(NUM_DETECTORS)]
    drift_cols = ["candidate_drift_%02d" % det for det in range(NUM_DETECTORS)]

    elem = df[elem_cols].to_numpy(dtype=np.float32)
    drift = df[drift_cols].to_numpy(dtype=np.float32)
    arr = np.zeros((len(df), NUM_DETECTORS, 2), dtype=np.float32)
    arr[:, :, 0] = elem
    arr[:, :, 1] = drift

    for start, stop in INACTIVE_SLICES:
        arr[:, start:stop, :] = 0.0

    return arr


def predict_by_charge(df, model_mup, model_mum, chunk_size):
    predictions = np.full((len(df), 3), np.nan, dtype=np.float32)

    for charge_name, model in [("mup", model_mup), ("mum", model_mum)]:
        idx = np.where(df["charge"].astype(str).to_numpy() == charge_name)[0]
        if len(idx) == 0:
            continue
        print("[predict] charge", charge_name, "tracks:", len(idx))
        sub = df.iloc[idx]
        x = build_momentum_input(sub)
        out_chunks = []
        for start in range(0, len(x), chunk_size):
            stop = min(start + chunk_size, len(x))
            pred = model.predict(x[start:stop], verbose=0)
            out_chunks.append(np.asarray(pred, dtype=np.float32))
            print("[predict]", charge_name, str(stop) + "/" + str(len(x)))
        pred_all = np.concatenate(out_chunks, axis=0)
        if pred_all.shape[1] != 3:
            raise RuntimeError("Momentum model output must have shape (N, 3), got " + str(pred_all.shape))
        predictions[idx, :] = pred_all

    return predictions


def add_error_columns(df, pred):
    result = df.copy()
    result["pred_px"] = pred[:, 0]
    result["pred_py"] = pred[:, 1]
    result["pred_pz"] = pred[:, 2]

    true_vec = result[["true_px", "true_py", "true_pz"]].to_numpy(dtype=np.float64)
    pred_vec = result[["pred_px", "pred_py", "pred_pz"]].to_numpy(dtype=np.float64)
    diff = pred_vec - true_vec

    true_p = np.linalg.norm(true_vec, axis=1)
    pred_p = np.linalg.norm(pred_vec, axis=1)
    l2 = np.linalg.norm(diff, axis=1)
    denom = np.where(true_p > 0.0, true_p, np.nan)
    true_pz_abs = np.where(np.abs(true_vec[:, 2]) > 0.0, np.abs(true_vec[:, 2]), np.nan)

    result["momentum_l2_error"] = l2
    result["momentum_relative_l2_error"] = l2 / denom
    result["momentum_abs_mag_error"] = np.abs(pred_p - true_p)
    result["momentum_relative_mag_error"] = np.abs(pred_p - true_p) / denom
    result["abs_px_error"] = np.abs(diff[:, 0])
    result["abs_py_error"] = np.abs(diff[:, 1])
    result["abs_pz_error"] = np.abs(diff[:, 2])
    result["relative_pz_error"] = np.abs(diff[:, 2]) / true_pz_abs
    return result


def summarize_thresholds(df, thresholds, score_col):
    rows = []
    total = len(df)
    valid = df[np.isfinite(df["momentum_relative_l2_error"])]
    for threshold in thresholds:
        selected = valid[valid[score_col] >= threshold]
        row = {
            "threshold": threshold,
            "n_kept_tracks": int(len(selected)),
            "kept_track_fraction": float(len(selected)) / float(total) if total > 0 else 0.0,
        }
        for col in [
            "momentum_l2_error",
            "momentum_relative_l2_error",
            "momentum_abs_mag_error",
            "momentum_relative_mag_error",
            "abs_pz_error",
            "relative_pz_error",
            "mean_abs_residual_on_truth_hits",
            "exact_fraction_on_truth_hits",
            "residual_leq_2_fraction_on_truth_hits",
        ]:
            if col in selected.columns and len(selected) > 0:
                values = selected[col].to_numpy(dtype=np.float64)
                row[col + "_mean"] = float(np.nanmean(values))
                row[col + "_median"] = float(np.nanmedian(values))
                row[col + "_p95"] = float(np.nanpercentile(values, 95))
            else:
                row[col + "_mean"] = float("nan")
                row[col + "_median"] = float("nan")
                row[col + "_p95"] = float("nan")
        if len(selected) > 0:
            row["bad_rel_l2_gt_0p10_fraction"] = float(np.mean(selected["momentum_relative_l2_error"].to_numpy(dtype=np.float64) > 0.10))
            row["bad_rel_l2_gt_0p20_fraction"] = float(np.mean(selected["momentum_relative_l2_error"].to_numpy(dtype=np.float64) > 0.20))
        else:
            row["bad_rel_l2_gt_0p10_fraction"] = float("nan")
            row["bad_rel_l2_gt_0p20_fraction"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_charge(df, thresholds, score_col):
    rows = []
    for charge in sorted(df["charge"].astype(str).unique()):
        sub = df[df["charge"].astype(str) == charge]
        tmp = summarize_thresholds(sub, thresholds, score_col)
        tmp.insert(0, "charge", charge)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_summary_text(path, input_csv, df, threshold_df, score_col):
    with open(path, "w") as fout:
        fout.write("QMetric momentum threshold summary\n")
        fout.write("=================================\n")
        fout.write("input_csv: " + str(input_csv) + "\n")
        fout.write("rows: " + str(len(df)) + "\n")
        fout.write("score_col: " + str(score_col) + "\n")
        fout.write("charges: " + str(df["charge"].value_counts().to_dict()) + "\n\n")
        fout.write("Overall momentum error\n")
        fout.write("----------------------\n")
        for col in ["momentum_l2_error", "momentum_relative_l2_error", "momentum_relative_mag_error", "relative_pz_error"]:
            values = df[col].to_numpy(dtype=np.float64)
            fout.write(col + ": mean=" + str(float(np.nanmean(values))) + ", median=" + str(float(np.nanmedian(values))) + "\n")
        fout.write("\nThreshold sweep preview\n")
        fout.write("-----------------------\n")
        preview_cols = [
            "threshold",
            "n_kept_tracks",
            "kept_track_fraction",
            "momentum_relative_l2_error_mean",
            "momentum_relative_l2_error_median",
            "momentum_relative_mag_error_mean",
            "relative_pz_error_mean",
            "bad_rel_l2_gt_0p10_fraction",
            "mean_abs_residual_on_truth_hits_mean",
        ]
        fout.write(threshold_df[preview_cols].to_string(index=False))
        fout.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate momentum reconstruction after QMetric track selection.")
    parser.add_argument("input_csv", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs/qmetric_momentum_summary")
    parser.add_argument("--score_col", type=str, default="qmetric_score_v0")
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--mup_model", type=str, default="/mnt/code/checkpoints/mom_mup.h5")
    parser.add_argument("--mum_model", type=str, default="/mnt/code/checkpoints/mom_mum.h5")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--write_candidate_predictions", type=int, default=0)
    parser.add_argument("--print_env", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_env:
        print_environment()

    thresholds = parse_thresholds(args.thresholds)
    print("[data] reading:", args.input_csv)
    df = pd.read_csv(args.input_csv)
    print("[data] rows:", len(df))
    check_required_columns(df, args.score_col)

    model_mup = load_momentum_model(args.mup_model)
    model_mum = load_momentum_model(args.mum_model)
    pred = predict_by_charge(df, model_mup, model_mum, args.chunk_size)
    result = add_error_columns(df, pred)

    os.makedirs(args.output_dir, exist_ok=True)
    threshold_df = summarize_thresholds(result, thresholds, args.score_col)
    threshold_by_charge = summarize_by_charge(result, thresholds, args.score_col)

    threshold_path = os.path.join(args.output_dir, "momentum_threshold_sweep.csv")
    charge_path = os.path.join(args.output_dir, "momentum_threshold_sweep_by_charge.csv")
    summary_path = os.path.join(args.output_dir, "momentum_summary.txt")
    threshold_df.to_csv(threshold_path, index=False)
    threshold_by_charge.to_csv(charge_path, index=False)
    write_summary_text(summary_path, args.input_csv, result, threshold_df, args.score_col)

    if args.write_candidate_predictions:
        result.to_csv(os.path.join(args.output_dir, "momentum_candidate_predictions.csv"), index=False)

    print("[done] wrote:", threshold_path)
    print("[done] wrote:", charge_path)
    print("[done] wrote:", summary_path)


if __name__ == "__main__":
    main()
