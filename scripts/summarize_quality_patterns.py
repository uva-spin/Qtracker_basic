#!/usr/bin/env python3
"""
Summarize QMetric candidate-level pattern tables.

Input:
    CSV created by scripts/build_quality_pattern_table.py

Outputs:
    threshold_sweep.csv
    threshold_sweep_by_charge.csv
    quantile_summary.csv
    feature_correlations.csv
    summary.txt
    optional PNG plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_thresholds(text):
    if text is None or text.strip() == "":
        return [
            0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.92, 0.94, 0.95, 0.96, 0.97,
            0.98, 0.99, 0.995,
        ]

    values = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        values.append(float(part))
    return values


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def safe_mean(frame, column):
    if column not in frame.columns or len(frame) == 0:
        return np.nan
    return float(frame[column].mean())


def safe_median(frame, column):
    if column not in frame.columns or len(frame) == 0:
        return np.nan
    return float(frame[column].median())


def safe_quantile(frame, column, q):
    if column not in frame.columns or len(frame) == 0:
        return np.nan
    return float(frame[column].quantile(q))


def summarize_frame(frame, total_rows, threshold=None, charge=None):
    row = {
        "threshold": threshold,
        "charge": charge if charge is not None else "all",
        "n_kept": int(len(frame)),
        "kept_fraction": float(len(frame) / total_rows) if total_rows > 0 else np.nan,
        "mean_abs_residual": safe_mean(frame, "mean_abs_residual_on_truth_hits"),
        "median_abs_residual": safe_median(frame, "mean_abs_residual_on_truth_hits"),
        "q25_abs_residual": safe_quantile(frame, "mean_abs_residual_on_truth_hits", 0.25),
        "q75_abs_residual": safe_quantile(frame, "mean_abs_residual_on_truth_hits", 0.75),
        "mean_max_abs_residual": safe_mean(frame, "max_abs_residual_on_truth_hits"),
        "exact_fraction": safe_mean(frame, "exact_fraction_on_truth_hits"),
        "residual_leq_2_fraction": safe_mean(frame, "residual_leq_2_fraction_on_truth_hits"),
        "mean_softmax_conf": safe_mean(frame, "softmax_conf_mean"),
        "mean_softmax_margin": safe_mean(frame, "softmax_margin_mean"),
        "mean_softmax_entropy": safe_mean(frame, "softmax_entropy_mean"),
        "mean_missing_fraction": safe_mean(frame, "missing_fraction_active"),
        "mean_event_active_occupancy": safe_mean(frame, "event_active_occupancy"),
        "mean_local_density": safe_mean(frame, "local_density_mean"),
    }

    if "residual_leq_1_count" in frame.columns and "truth_hit_count_active" in frame.columns and len(frame) > 0:
        truth_count = frame["truth_hit_count_active"].replace(0, np.nan)
        row["residual_leq_1_fraction"] = float((frame["residual_leq_1_count"] / truth_count).mean())
    else:
        row["residual_leq_1_fraction"] = np.nan

    return row


def build_threshold_sweep(df, score_col, thresholds):
    rows = []
    total_rows = len(df)
    for threshold in thresholds:
        selected = df[df[score_col] >= threshold]
        rows.append(summarize_frame(selected, total_rows, threshold=threshold))
    return pd.DataFrame(rows)


def build_threshold_sweep_by_charge(df, score_col, thresholds):
    if "charge" not in df.columns:
        return pd.DataFrame()

    rows = []
    total_rows_by_charge = df.groupby("charge").size().to_dict()
    for threshold in thresholds:
        selected = df[df[score_col] >= threshold]
        for charge, frame in selected.groupby("charge", sort=True):
            total_rows = int(total_rows_by_charge.get(charge, len(df)))
            rows.append(summarize_frame(frame, total_rows, threshold=threshold, charge=charge))
    return pd.DataFrame(rows)


def build_quantile_summary(df, score_col, n_bins):
    if n_bins <= 1:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["score_quantile_bin"] = pd.qcut(
        tmp[score_col],
        q=n_bins,
        duplicates="drop",
    )

    rows = []
    total_rows = len(df)
    for bin_label, frame in tmp.groupby("score_quantile_bin", observed=True, sort=True):
        row = summarize_frame(frame, total_rows, threshold=None)
        row["score_bin"] = str(bin_label)
        row["score_min"] = float(frame[score_col].min())
        row["score_max"] = float(frame[score_col].max())
        row["score_mean"] = float(frame[score_col].mean())
        rows.append(row)

    columns = ["score_bin", "score_min", "score_max", "score_mean"]
    return pd.DataFrame(rows)[columns + [c for c in rows[0].keys() if c not in columns]] if rows else pd.DataFrame()


def build_feature_correlations(df, score_col, target_col):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric.columns:
        return pd.DataFrame()

    rows = []
    for col in numeric.columns:
        if col == target_col:
            continue
        valid = numeric[[col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 3:
            continue
        pearson = float(valid[col].corr(valid[target_col], method="pearson"))
        spearman = float(valid[col].corr(valid[target_col], method="spearman"))
        rows.append({
            "feature": col,
            "pearson_with_" + target_col: pearson,
            "spearman_with_" + target_col: spearman,
            "abs_pearson": abs(pearson) if not np.isnan(pearson) else np.nan,
            "abs_spearman": abs(spearman) if not np.isnan(spearman) else np.nan,
        })

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return result
    return result.sort_values("abs_pearson", ascending=False)


def write_summary_text(path, input_csv, df, score_col, threshold_sweep, quantile_summary, correlations):
    lines = []
    lines.append("QMetric pattern table summary")
    lines.append("================================")
    lines.append("input_csv: " + input_csv)
    lines.append("rows: " + str(len(df)))
    if "event_id" in df.columns:
        lines.append("events: " + str(df["event_id"].nunique()))
    if "charge" in df.columns:
        charge_counts = df["charge"].value_counts().to_dict()
        lines.append("charges: " + str(charge_counts))
    lines.append("score_col: " + score_col)
    lines.append("")

    lines.append("Overall residual quality")
    lines.append("------------------------")
    for col in [
        "mean_abs_residual_on_truth_hits",
        "exact_fraction_on_truth_hits",
        "residual_leq_2_fraction_on_truth_hits",
        "softmax_conf_mean",
        "softmax_entropy_mean",
        "event_active_occupancy",
        "local_density_mean",
        score_col,
    ]:
        if col in df.columns:
            lines.append(
                col + ": mean=" + format(float(df[col].mean()), ".6g")
                + ", median=" + format(float(df[col].median()), ".6g")
            )
    lines.append("")

    lines.append("Threshold sweep preview")
    lines.append("-----------------------")
    preview_cols = [
        "threshold", "n_kept", "kept_fraction", "mean_abs_residual",
        "exact_fraction", "residual_leq_2_fraction",
    ]
    if len(threshold_sweep) > 0:
        lines.append(threshold_sweep[preview_cols].to_string(index=False))
    lines.append("")

    lines.append("Score quantile summary")
    lines.append("----------------------")
    if len(quantile_summary) > 0:
        preview_cols = [
            "score_bin", "n_kept", "score_min", "score_max",
            "mean_abs_residual", "exact_fraction", "residual_leq_2_fraction",
        ]
        lines.append(quantile_summary[preview_cols].to_string(index=False))
    lines.append("")

    lines.append("Top feature correlations with mean_abs_residual_on_truth_hits")
    lines.append("-------------------------------------------------------------")
    if len(correlations) > 0:
        lines.append(correlations.head(20).to_string(index=False))
    lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def make_plots(df, score_col, threshold_sweep, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[warn] Could not import matplotlib; skipping plots: " + str(exc))
        return

    if len(threshold_sweep) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(threshold_sweep["threshold"], threshold_sweep["kept_fraction"], marker="o")
        plt.xlabel("QMetric threshold")
        plt.ylabel("Kept candidate fraction")
        plt.title("Candidate retention vs QMetric threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kept_fraction_vs_threshold.png"), dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(threshold_sweep["threshold"], threshold_sweep["mean_abs_residual"], marker="o")
        plt.xlabel("QMetric threshold")
        plt.ylabel("Mean absolute residual")
        plt.title("Residual quality vs QMetric threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_residual_vs_threshold.png"), dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(threshold_sweep["threshold"], threshold_sweep["residual_leq_2_fraction"], marker="o")
        plt.xlabel("QMetric threshold")
        plt.ylabel("Residual <= 2 fraction")
        plt.title("Within-2 hit fraction vs QMetric threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residual_leq_2_vs_threshold.png"), dpi=140)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df[score_col].dropna(), bins=60)
    plt.xlabel(score_col)
    plt.ylabel("Candidate count")
    plt.title("QMetric score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qmetric_score_histogram.png"), dpi=140)
    plt.close()

    if "mean_abs_residual_on_truth_hits" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[score_col], df["mean_abs_residual_on_truth_hits"], s=4, alpha=0.25)
        plt.xlabel(score_col)
        plt.ylabel("Mean absolute residual")
        plt.title("Residual vs QMetric score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residual_vs_qmetric_score.png"), dpi=140)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize QMetric pattern table CSV files.")
    parser.add_argument("input_csv", type=str, help="Path to qmetric pattern table CSV.")
    parser.add_argument("--output_dir", type=str, default="outputs/qmetric_summary", help="Directory for summary outputs.")
    parser.add_argument("--score_col", type=str, default="qmetric_score_v0", help="Score column to threshold.")
    parser.add_argument("--target_col", type=str, default="mean_abs_residual_on_truth_hits", help="Target column for correlations.")
    parser.add_argument("--thresholds", type=str, default="", help="Comma-separated thresholds. Empty uses default list.")
    parser.add_argument("--quantile_bins", type=int, default=4, help="Number of score quantile bins.")
    parser.add_argument("--plots", type=int, default=1, help="Set to 1 to write PNG plots.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("[load] " + args.input_csv)
    df = pd.read_csv(args.input_csv)
    require_columns(df, [args.score_col])

    thresholds = parse_thresholds(args.thresholds)

    print("[summary] rows: " + str(len(df)))
    threshold_sweep = build_threshold_sweep(df, args.score_col, thresholds)
    threshold_sweep_by_charge = build_threshold_sweep_by_charge(df, args.score_col, thresholds)
    quantile_summary = build_quantile_summary(df, args.score_col, args.quantile_bins)
    correlations = build_feature_correlations(df, args.score_col, args.target_col)

    threshold_path = os.path.join(args.output_dir, "threshold_sweep.csv")
    charge_path = os.path.join(args.output_dir, "threshold_sweep_by_charge.csv")
    quantile_path = os.path.join(args.output_dir, "quantile_summary.csv")
    corr_path = os.path.join(args.output_dir, "feature_correlations.csv")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    threshold_sweep.to_csv(threshold_path, index=False)
    threshold_sweep_by_charge.to_csv(charge_path, index=False)
    quantile_summary.to_csv(quantile_path, index=False)
    correlations.to_csv(corr_path, index=False)

    write_summary_text(
        summary_path,
        args.input_csv,
        df,
        args.score_col,
        threshold_sweep,
        quantile_summary,
        correlations,
    )

    if args.plots:
        make_plots(df, args.score_col, threshold_sweep, args.output_dir)

    print("[done] wrote " + threshold_path)
    print("[done] wrote " + charge_path)
    print("[done] wrote " + quantile_path)
    print("[done] wrote " + corr_path)
    print("[done] wrote " + summary_path)


if __name__ == "__main__":
    main()
