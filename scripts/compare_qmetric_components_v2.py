#!/usr/bin/env python3
"""
Compare QMetric component scores, including v2 candidates, against residual and momentum reconstruction quality.

This script is for QMetric development, not final production inference.
It evaluates candidate-track selection rules. One row is one candidate track.

Input can be either:
    1. A candidate table that already contains momentum prediction/error columns.
    2. A candidate table with candidate_elem_00..61, candidate_drift_00..61,
       true_px/true_py/true_pz columns plus momentum model paths.

Truth and momentum-error columns are calibration/evaluation targets only.
They must not be used as runtime QMetric inputs.
"""

import argparse
import math
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

NUM_DETECTORS = 62

MOMENTUM_MASK_SLICES = (
    (7, 12),
    (55, 58),
    (59, 62),
)

DEFAULT_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
]

DEFAULT_KEEP_FRACTIONS = [
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.25,
    0.20,
    0.10,
]

RUNTIME_FEATURE_COLUMNS = [
    "missing_fraction_active",
    "station_coverage_fraction",
    "max_missing_gap_active",
    "event_mean_layer_occupancy",
    "event_active_occupancy",
    "local_density_mean",
    "softmax_conf_mean",
    "softmax_conf_min",
    "softmax_margin_mean",
    "softmax_margin_min",
    "softmax_entropy_mean",
    "presence_prob_mean",
    "mean_abs_element_step",
    "max_abs_element_step",
]

TARGET_COLUMNS = [
    "mean_abs_residual_on_truth_hits",
    "exact_fraction_on_truth_hits",
    "residual_leq_2_fraction_on_truth_hits",
    "momentum_relative_l2_error",
    "momentum_relative_mag_error",
    "relative_pz_error",
]


def parse_float_list(text, default_values):
    if text is None or str(text).strip() == "":
        return list(default_values)
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def print_environment():
    print("[env] python:", sys.executable)
    print("[env] numpy:", np.__version__, np.__file__)
    print("[env] pandas:", pd.__version__, pd.__file__)
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


def clip01(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def get_col(df, name, default_value=0.0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default_value).to_numpy(dtype=np.float64)
    return np.full(len(df), float(default_value), dtype=np.float64)


def add_component_scores(df):
    missing = clip01(get_col(df, "missing_fraction_active"))
    station = clip01(get_col(df, "station_coverage_fraction"))
    gap = clip01(get_col(df, "max_missing_gap_active") / 20.0)
    occ = clip01(get_col(df, "event_mean_layer_occupancy") / 20.0)
    local = clip01(get_col(df, "local_density_mean") / 5.0)
    conf = clip01(get_col(df, "softmax_conf_mean"))
    conf_min = clip01(get_col(df, "softmax_conf_min"))
    margin = clip01(get_col(df, "softmax_margin_mean"))
    margin_min = clip01(get_col(df, "softmax_margin_min"))
    entropy = clip01(get_col(df, "softmax_entropy_mean"))
    step = clip01(get_col(df, "mean_abs_element_step") / 100.0)
    max_step = clip01(get_col(df, "max_abs_element_step") / 200.0)

    shape = 1.0 - 0.60 * missing - 0.25 * (1.0 - station) - 0.15 * gap
    softmax = 0.40 * conf + 0.25 * conf_min + 0.25 * margin + 0.10 * margin_min - 0.20 * entropy + 0.20
    occupancy = 1.0 - 0.60 * occ - 0.40 * local
    smoothness = 1.0 - 0.55 * step - 0.45 * max_step

    df["score_shape_only"] = clip01(shape)
    df["score_softmax_only"] = clip01(softmax)
    df["score_occupancy_only"] = clip01(occupancy)
    df["score_smoothness_only"] = clip01(smoothness)

    df["score_v0_no_occupancy"] = clip01(
        1.0
        - 0.55 * missing
        - 0.20 * (1.0 - station)
        - 0.10 * gap
        + 0.15 * conf
        + 0.15 * margin
        - 0.15 * entropy
    )
    df["score_v0_no_softmax"] = clip01(
        1.0
        - 0.55 * missing
        - 0.20 * (1.0 - station)
        - 0.10 * gap
        - 0.10 * occ
    )
    df["score_v0_no_shape"] = clip01(
        0.70
        - 0.10 * occ
        + 0.15 * conf
        + 0.15 * margin
        - 0.15 * entropy
    )
    df["score_v1_balanced"] = clip01(
        0.50 * df["score_softmax_only"].to_numpy(dtype=np.float64)
        + 0.25 * df["score_shape_only"].to_numpy(dtype=np.float64)
        + 0.15 * df["score_occupancy_only"].to_numpy(dtype=np.float64)
        + 0.10 * df["score_smoothness_only"].to_numpy(dtype=np.float64)
    )

    # v2 candidates: keep v0's interpretable structure, but test whether
    # event-level occupancy was over-penalized for momentum selection.
    # These are still development/calibration scores, not final production scores.
    df["score_v2_low_occupancy"] = clip01(
        1.0
        - 0.55 * missing
        - 0.20 * (1.0 - station)
        - 0.10 * gap
        - 0.04 * occ
        + 0.15 * conf
        + 0.15 * margin
        - 0.15 * entropy
    )
    df["score_v2_local_density"] = clip01(
        1.0
        - 0.55 * missing
        - 0.20 * (1.0 - station)
        - 0.10 * gap
        - 0.06 * local
        + 0.15 * conf
        + 0.15 * margin
        - 0.15 * entropy
    )
    df["score_v2_low_occ_local"] = clip01(
        1.0
        - 0.55 * missing
        - 0.20 * (1.0 - station)
        - 0.10 * gap
        - 0.03 * occ
        - 0.04 * local
        + 0.15 * conf
        + 0.15 * margin
        - 0.15 * entropy
    )
    df["score_v2_softmax_min_guard"] = clip01(
        1.0
        - 0.52 * missing
        - 0.18 * (1.0 - station)
        - 0.09 * gap
        - 0.03 * occ
        - 0.04 * local
        + 0.12 * conf
        + 0.05 * conf_min
        + 0.10 * margin
        + 0.04 * margin_min
        - 0.15 * entropy
    )
    df["score_v2_softmax_shape_local"] = clip01(
        1.0
        - 0.50 * missing
        - 0.20 * (1.0 - station)
        - 0.08 * gap
        - 0.05 * local
        + 0.18 * conf
        + 0.12 * margin
        - 0.16 * entropy
    )

    if "qmetric_score_v0" not in df.columns:
        df["qmetric_score_v0"] = clip01(
            1.0
            - 0.55 * missing
            - 0.20 * (1.0 - station)
            - 0.10 * gap
            - 0.10 * occ
            + 0.15 * conf
            + 0.15 * margin
            - 0.15 * entropy
        )

    return df


def has_momentum_errors(df):
    required = [
        "pred_px",
        "pred_py",
        "pred_pz",
        "momentum_l2_error",
        "momentum_relative_l2_error",
        "momentum_abs_mag_error",
        "momentum_relative_mag_error",
    ]
    return all(col in df.columns for col in required)


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(label + " missing required columns: " + ", ".join(missing[:20]))


def load_momentum_model(model_path):
    import tensorflow as tf

    print("[momentum] Loading model:", model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        try:
            from tensorflow.keras.losses import MeanSquaredError
            return tf.keras.models.load_model(
                model_path,
                custom_objects={"mse": MeanSquaredError()},
                compile=False,
            )
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to load momentum model: " + str(model_path)
                + "\nFirst error: " + str(exc)
                + "\nSecond error: " + str(second_exc)
            )


def build_momentum_input(df):
    elem_cols = ["candidate_elem_" + f"{idx:02d}" for idx in range(NUM_DETECTORS)]
    drift_cols = ["candidate_drift_" + f"{idx:02d}" for idx in range(NUM_DETECTORS)]
    require_columns(df, elem_cols + drift_cols, "candidate table")

    elems = df[elem_cols].to_numpy(dtype=np.float32)
    drifts = df[drift_cols].to_numpy(dtype=np.float32)
    tracks = np.stack([elems, drifts], axis=-1)
    return tracks


def apply_momentum_mask(tracks):
    masked = np.array(tracks, copy=True)
    for start, stop in MOMENTUM_MASK_SLICES:
        masked[:, start:stop, :] = 0.0
    return masked


def predict_by_chunks(model, tracks, chunk_size):
    preds = []
    for start in range(0, len(tracks), chunk_size):
        stop = min(start + chunk_size, len(tracks))
        pred = model.predict(tracks[start:stop], verbose=0)
        preds.append(np.asarray(pred, dtype=np.float32))
        print("[momentum] predicted", str(stop) + "/" + str(len(tracks)), "tracks")
    if not preds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def add_momentum_predictions(df, args):
    require_columns(df, ["charge", "true_px", "true_py", "true_pz"], "candidate table")

    tracks = build_momentum_input(df)
    if args.apply_momentum_mask:
        tracks = apply_momentum_mask(tracks)

    pred = np.zeros((len(df), 3), dtype=np.float32)
    charges = df["charge"].astype(str).to_numpy()

    mup_mask = charges == "mup"
    mum_mask = charges == "mum"

    if np.any(mup_mask):
        mup_model = load_momentum_model(args.mup_model)
        pred[mup_mask] = predict_by_chunks(mup_model, tracks[mup_mask], args.chunk_size)
    if np.any(mum_mask):
        mum_model = load_momentum_model(args.mum_model)
        pred[mum_mask] = predict_by_chunks(mum_model, tracks[mum_mask], args.chunk_size)

    true = df[["true_px", "true_py", "true_pz"]].to_numpy(dtype=np.float64)
    pred64 = pred.astype(np.float64)
    true_p = np.linalg.norm(true, axis=1)
    pred_p = np.linalg.norm(pred64, axis=1)
    den = np.maximum(true_p, 1e-8)
    pz_den = np.maximum(np.abs(true[:, 2]), 1e-8)

    diff = pred64 - true
    df["pred_px"] = pred64[:, 0]
    df["pred_py"] = pred64[:, 1]
    df["pred_pz"] = pred64[:, 2]
    df["momentum_l2_error"] = np.linalg.norm(diff, axis=1)
    df["momentum_relative_l2_error"] = df["momentum_l2_error"].to_numpy(dtype=np.float64) / den
    df["momentum_abs_mag_error"] = np.abs(pred_p - true_p)
    df["momentum_relative_mag_error"] = df["momentum_abs_mag_error"].to_numpy(dtype=np.float64) / den
    df["abs_px_error"] = np.abs(diff[:, 0])
    df["abs_py_error"] = np.abs(diff[:, 1])
    df["abs_pz_error"] = np.abs(diff[:, 2])
    df["relative_pz_error"] = df["abs_pz_error"].to_numpy(dtype=np.float64) / pz_den
    return df


def metric_summary(subset):
    n = len(subset)
    result = {
        "n_kept": int(n),
    }
    if n == 0:
        return result

    metrics = [
        "mean_abs_residual_on_truth_hits",
        "exact_fraction_on_truth_hits",
        "residual_leq_2_fraction_on_truth_hits",
        "momentum_relative_l2_error",
        "momentum_relative_mag_error",
        "relative_pz_error",
    ]
    for col in metrics:
        if col in subset.columns:
            vals = pd.to_numeric(subset[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                result[col + "_mean"] = float(np.mean(vals))
                result[col + "_median"] = float(np.median(vals))
                result[col + "_p90"] = float(np.quantile(vals, 0.90))
                result[col + "_p95"] = float(np.quantile(vals, 0.95))

    if "momentum_relative_l2_error" in subset.columns:
        vals = pd.to_numeric(subset["momentum_relative_l2_error"], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            result["bad_rel_l2_gt_0p05_fraction"] = float(np.mean(vals > 0.05))
            result["bad_rel_l2_gt_0p10_fraction"] = float(np.mean(vals > 0.10))
            result["bad_rel_l2_gt_0p20_fraction"] = float(np.mean(vals > 0.20))
    return result


def build_threshold_sweep(df, score_cols, thresholds, by_charge=False):
    rows = []
    total = len(df)
    groups = [("all", df)]
    if by_charge and "charge" in df.columns:
        groups = [(str(charge), group) for charge, group in df.groupby("charge")]

    for group_name, group in groups:
        group_total = len(group)
        for score_col in score_cols:
            scores = pd.to_numeric(group[score_col], errors="coerce")
            for threshold in thresholds:
                selected = group[scores >= threshold]
                row = {
                    "group": group_name,
                    "score_name": score_col,
                    "threshold": float(threshold),
                    "n_total": int(group_total),
                    "kept_fraction": float(len(selected) / group_total) if group_total else 0.0,
                    "kept_fraction_global": float(len(selected) / total) if total else 0.0,
                }
                row.update(metric_summary(selected))
                rows.append(row)
    return pd.DataFrame(rows)


def build_efficiency_sweep(df, score_cols, keep_fractions, by_charge=False):
    rows = []
    total = len(df)
    groups = [("all", df)]
    if by_charge and "charge" in df.columns:
        groups = [(str(charge), group) for charge, group in df.groupby("charge")]

    for group_name, group in groups:
        group_total = len(group)
        for score_col in score_cols:
            scores = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=np.float64)
            scores = np.nan_to_num(scores, nan=-np.inf)
            for keep_fraction in keep_fractions:
                keep_fraction = float(keep_fraction)
                if group_total == 0:
                    threshold = math.nan
                    selected = group.iloc[0:0]
                else:
                    threshold = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - keep_fraction))))
                    selected = group[scores >= threshold]
                row = {
                    "group": group_name,
                    "score_name": score_col,
                    "target_keep_fraction": keep_fraction,
                    "threshold": threshold,
                    "n_total": int(group_total),
                    "kept_fraction": float(len(selected) / group_total) if group_total else 0.0,
                    "kept_fraction_global": float(len(selected) / total) if total else 0.0,
                }
                row.update(metric_summary(selected))
                rows.append(row)
    return pd.DataFrame(rows)


def build_correlations(df, score_cols):
    rows = []
    for score_col in score_cols:
        for target_col in TARGET_COLUMNS:
            if target_col not in df.columns:
                continue
            pair = df[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                continue
            pearson = pair[score_col].corr(pair[target_col], method="pearson")
            spearman = pair[score_col].corr(pair[target_col], method="spearman")
            rows.append({
                "score_name": score_col,
                "target": target_col,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "abs_pearson": float(abs(pearson)),
                "abs_spearman": float(abs(spearman)),
            })
    return pd.DataFrame(rows)


def write_summary(output_path, df, score_cols, threshold_sweep, efficiency_sweep, correlations):
    with open(output_path, "w") as fout:
        fout.write("QMetric component comparison summary\n")
        fout.write("====================================\n")
        fout.write("rows: " + str(len(df)) + "\n")
        if "event_id" in df.columns:
            fout.write("events: " + str(df["event_id"].nunique()) + "\n")
        if "charge" in df.columns:
            fout.write("charges: " + str(df["charge"].value_counts().to_dict()) + "\n")
        fout.write("\nScore columns\n-------------\n")
        for col in score_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            fout.write(
                f"{col}: mean={vals.mean():.6g}, median={vals.median():.6g}, "
                f"min={vals.min():.6g}, max={vals.max():.6g}\n"
            )

        fout.write("\nAt approximately 50% kept tracks\n--------------------------------\n")
        if len(efficiency_sweep) > 0:
            eff50 = efficiency_sweep[
                (efficiency_sweep["group"] == "all")
                & (np.isclose(efficiency_sweep["target_keep_fraction"].astype(float), 0.50))
            ].copy()
            sort_col = "momentum_relative_l2_error_mean"
            if sort_col in eff50.columns:
                eff50 = eff50.sort_values(sort_col)
            display_cols = [
                "score_name",
                "threshold",
                "kept_fraction",
                "mean_abs_residual_on_truth_hits_mean",
                "momentum_relative_l2_error_mean",
                "momentum_relative_l2_error_median",
                "bad_rel_l2_gt_0p10_fraction",
            ]
            display_cols = [col for col in display_cols if col in eff50.columns]
            fout.write(eff50[display_cols].to_string(index=False))
            fout.write("\n")

        fout.write("\nAt fixed threshold >= 0.94\n-------------------------\n")
        if len(threshold_sweep) > 0:
            fixed = threshold_sweep[
                (threshold_sweep["group"] == "all")
                & (np.isclose(threshold_sweep["threshold"].astype(float), 0.94))
            ].copy()
            sort_col = "momentum_relative_l2_error_mean"
            if sort_col in fixed.columns:
                fixed = fixed.sort_values(sort_col)
            display_cols = [
                "score_name",
                "kept_fraction",
                "mean_abs_residual_on_truth_hits_mean",
                "momentum_relative_l2_error_mean",
                "momentum_relative_l2_error_median",
                "bad_rel_l2_gt_0p10_fraction",
            ]
            display_cols = [col for col in display_cols if col in fixed.columns]
            fout.write(fixed[display_cols].to_string(index=False))
            fout.write("\n")

        fout.write("\nCorrelations with momentum_relative_l2_error\n--------------------------------------------\n")
        if len(correlations) > 0:
            corr = correlations[correlations["target"] == "momentum_relative_l2_error"].copy()
            corr = corr.sort_values("abs_spearman", ascending=False)
            fout.write(corr.to_string(index=False))
            fout.write("\n")


def make_plots(output_dir, efficiency_sweep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[plot] matplotlib unavailable:", exc)
        return

    if len(efficiency_sweep) == 0:
        return
    plot_df = efficiency_sweep[efficiency_sweep["group"] == "all"].copy()
    if "momentum_relative_l2_error_mean" not in plot_df.columns:
        return

    plt.figure(figsize=(10, 6))
    for score_name, group in plot_df.groupby("score_name"):
        group = group.sort_values("kept_fraction")
        plt.plot(group["kept_fraction"], group["momentum_relative_l2_error_mean"], marker="o", label=score_name)
    plt.xlabel("Kept candidate-track fraction")
    plt.ylabel("Mean momentum relative L2 error")
    plt.title("QMetric component comparison at matched kept fraction")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "component_momentum_error_vs_kept_fraction.png"), dpi=140)
    plt.close()

    if "bad_rel_l2_gt_0p10_fraction" in plot_df.columns:
        plt.figure(figsize=(10, 6))
        for score_name, group in plot_df.groupby("score_name"):
            group = group.sort_values("kept_fraction")
            plt.plot(group["kept_fraction"], group["bad_rel_l2_gt_0p10_fraction"], marker="o", label=score_name)
        plt.xlabel("Kept candidate-track fraction")
        plt.ylabel("Fraction with relative L2 error > 0.10")
        plt.title("Momentum outlier rejection at matched kept fraction")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "component_bad_tail_vs_kept_fraction.png"), dpi=140)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare QMetric component scores for track-candidate selection.")
    parser.add_argument("input_csv", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs/qmetric_component_comparison")
    parser.add_argument("--thresholds", type=str, default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--keep_fractions", type=str, default=",".join(str(x) for x in DEFAULT_KEEP_FRACTIONS))
    parser.add_argument("--mup_model", type=str, default="")
    parser.add_argument("--mum_model", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--apply_momentum_mask", type=int, default=1)
    parser.add_argument("--write_candidate_predictions", type=int, default=0)
    parser.add_argument("--plots", type=int, default=1)
    parser.add_argument("--print_env", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.apply_momentum_mask = bool(args.apply_momentum_mask)
    args.write_candidate_predictions = bool(args.write_candidate_predictions)
    args.plots = bool(args.plots)

    if args.print_env:
        print_environment()

    thresholds = parse_float_list(args.thresholds, DEFAULT_THRESHOLDS)
    keep_fractions = parse_float_list(args.keep_fractions, DEFAULT_KEEP_FRACTIONS)

    print("[data] Reading:", args.input_csv)
    df = pd.read_csv(args.input_csv)
    print("[data] rows:", len(df))
    print("[data] columns:", len(df.columns))

    if not has_momentum_errors(df):
        if not args.mup_model or not args.mum_model:
            raise ValueError(
                "Input CSV does not contain momentum prediction/error columns. "
                "Provide --mup_model and --mum_model, or use a candidate_predictions CSV."
            )
        df = add_momentum_predictions(df, args)
    else:
        print("[momentum] Input already contains momentum prediction/error columns.")

    df = add_component_scores(df)
    score_cols = [
        "qmetric_score_v0",
        "score_v2_low_occupancy",
        "score_v2_local_density",
        "score_v2_low_occ_local",
        "score_v2_softmax_min_guard",
        "score_v2_softmax_shape_local",
        "score_v1_balanced",
        "score_softmax_only",
        "score_shape_only",
        "score_occupancy_only",
        "score_smoothness_only",
        "score_v0_no_occupancy",
        "score_v0_no_softmax",
        "score_v0_no_shape",
    ]
    score_cols = [col for col in score_cols if col in df.columns]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    threshold_sweep = build_threshold_sweep(df, score_cols, thresholds, by_charge=False)
    threshold_sweep_by_charge = build_threshold_sweep(df, score_cols, thresholds, by_charge=True)
    efficiency_sweep = build_efficiency_sweep(df, score_cols, keep_fractions, by_charge=False)
    efficiency_sweep_by_charge = build_efficiency_sweep(df, score_cols, keep_fractions, by_charge=True)
    correlations = build_correlations(df, score_cols)

    threshold_sweep.to_csv(os.path.join(output_dir, "component_threshold_sweep.csv"), index=False)
    threshold_sweep_by_charge.to_csv(os.path.join(output_dir, "component_threshold_sweep_by_charge.csv"), index=False)
    efficiency_sweep.to_csv(os.path.join(output_dir, "component_efficiency_sweep.csv"), index=False)
    efficiency_sweep_by_charge.to_csv(os.path.join(output_dir, "component_efficiency_sweep_by_charge.csv"), index=False)
    correlations.to_csv(os.path.join(output_dir, "component_correlations.csv"), index=False)

    if args.write_candidate_predictions:
        df.to_csv(os.path.join(output_dir, "component_candidate_predictions.csv"), index=False)

    write_summary(
        os.path.join(output_dir, "component_summary.txt"),
        df,
        score_cols,
        threshold_sweep,
        efficiency_sweep,
        correlations,
    )

    if args.plots:
        make_plots(output_dir, efficiency_sweep)

    print("[done] Wrote component comparison outputs to", output_dir)


if __name__ == "__main__":
    main()
