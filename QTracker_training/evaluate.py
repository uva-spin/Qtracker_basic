# ruff: noqa: E402

import os

import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity("error")

import argparse

import matplotlib.pyplot as plt
import numpy as np
import QTracker
import ROOT  # noqa: F401
import tensorflow as tf

# core TrackFinder loaders / custom loss
from models import data_loader
from models.layers import AxialAttention
from refine import refine_hit_arrays


def _detect_confidence_head(model):
    """Return True if the model has a confidence head (3+ outputs)."""
    if isinstance(model.output, (list, tuple)):
        return len(model.output) >= 3
    return False


def _compute_f1_per_event(y_pred_argmax, y_true, mu_idx):
    """Compute per-event F1 between predicted and true element-IDs for one muon.

    Args:
        y_pred_argmax: (N, 62) predicted element-IDs.
        y_true: (N, 62) ground-truth element-IDs.
        mu_idx: 0 for μ+, 1 for μ- (only used for labelling, not logic).

    Returns:
        f1: (N,) per-event F1 scores.
    """
    pred_nz = y_pred_argmax != 0
    true_nz = y_true != 0
    match = (y_pred_argmax == y_true) & pred_nz & true_nz

    tp = match.sum(axis=1).astype(np.float64)
    pred_pos = pred_nz.sum(axis=1).astype(np.float64)
    true_pos = true_nz.sum(axis=1).astype(np.float64)

    precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
    recall = np.where(true_pos > 0, tp / true_pos, 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    return f1


def _evaluate_confidence(
    confidence_scores, y_p_raw, y_m_raw, y_p_true, y_m_true, model_path
):
    """Evaluate the confidence head and print / plot metrics.

    Works for both Proposal A (event-level stop-or-go) and Proposal B
    (track-quality F1 overlap).  It reports:
      * Binary accuracy, precision, recall, F1 (threshold = 0.5)
      * Correlation between confidence and combined F1 overlap
      * Scatter plot of confidence vs F1 overlap
    """

    conf = confidence_scores.ravel()  # (N,)
    n_events = len(conf)

    # ---------- F1 overlap (Proposal-B style soft target) ----------
    f1_plus = _compute_f1_per_event(y_p_raw, y_p_true, mu_idx=0)
    f1_minus = _compute_f1_per_event(y_m_raw, y_m_true, mu_idx=1)
    f1_combined = (f1_plus + f1_minus) / 2.0  # mean over muon charges

    # ---------- Binary classification (Proposal-A style) ----------
    # A "good" event is one where the combined F1 >= 0.5
    # (also works if the GT simply has no tracks → F1 = 0)
    gt_binary = (f1_combined >= 0.5).astype(np.float32)
    pred_binary = (conf >= 0.5).astype(np.float32)

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    accuracy = (tp + tn) / n_events if n_events > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_cls = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("Confidence Head Evaluation")
    print("=" * 60)
    print(f"  Events evaluated     : {n_events}")
    print(f"  Mean confidence      : {np.mean(conf):.4f}")
    print(f"  Std  confidence      : {np.std(conf):.4f}")
    print(f"  Mean F1 overlap      : {np.mean(f1_combined):.4f}")
    print()
    print("  --- Binary metrics (threshold = 0.5) ---")
    print(f"  True Positives       : {tp}")
    print(f"  True Negatives       : {tn}")
    print(f"  False Positives      : {fp}")
    print(f"  False Negatives      : {fn}")
    print(f"  Accuracy             : {accuracy:.4f}")
    print(f"  Precision            : {precision:.4f}")
    print(f"  Recall               : {recall:.4f}")
    print(f"  F1 (classification)  : {f1_cls:.4f}")

    # ---------- Correlation ----------
    if np.std(conf) > 1e-8 and np.std(f1_combined) > 1e-8:
        corr = np.corrcoef(conf, f1_combined)[0, 1]
    else:
        corr = float("nan")
    print(f"\n  Pearson correlation (confidence vs F1 overlap): {corr:.4f}")

    # --- Residual correlation ---
    abs_res_p = np.mean(np.abs(y_p_true - y_p_raw), axis=1)
    abs_res_m = np.mean(np.abs(y_m_true - y_m_raw), axis=1)
    mean_abs_res = (abs_res_p + abs_res_m) / 2.0

    if np.std(conf) > 1e-8 and np.std(mean_abs_res) > 1e-8:
        corr_res = np.corrcoef(conf, mean_abs_res)[0, 1]
    else:
        corr_res = float("nan")
    print(f"  Pearson correlation (confidence vs mean |residual|): {corr_res:.4f}")

    # ---------- Confidence vs F1 scatter plot ----------
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: confidence vs F1 overlap
    ax = axes[0]
    ax.scatter(f1_combined, conf, s=4, alpha=0.3, edgecolors="none")
    ax.set_xlabel("F1 Overlap (pred vs true)")
    ax.set_ylabel("Confidence Score")
    ax.set_title(f"Confidence vs F1 Overlap  (ρ = {corr:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y = x")
    ax.legend()

    # Panel 2: confidence vs mean |residual|
    ax = axes[1]
    ax.scatter(mean_abs_res, conf, s=4, alpha=0.3, edgecolors="none")
    ax.set_xlabel("Mean Absolute Residual")
    ax.set_ylabel("Confidence Score")
    ax.set_title(f"Confidence vs |Residual|  (ρ = {corr_res:.3f})")

    plt.tight_layout()
    base = os.path.splitext(os.path.basename(model_path))[0]
    fname = f"{base}_confidence_eval.png"
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()
    print(f"\n  Saved confidence plot to plots/{fname}")
    print("=" * 60)


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
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
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
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(args.root_file)
    if X_test is None:
        return

    y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=1)
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

    has_confidence = _detect_confidence_head(model)
    if has_confidence:
        print("\n[INFO] Model has a confidence head (3 outputs).")
    else:
        print("\n[INFO] Model has no confidence head (2 outputs).")

    seg_preds = []
    conf_preds = [] if has_confidence else None
    chunk_size = 128

    for i in range(0, len(X_test), chunk_size):
        X_chunk = tf.cast(X_test[i : i + chunk_size], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        seg_preds.append(y_chunk[1])  # segment output is always index 1
        if has_confidence:
            conf_preds.append(y_chunk[2])  # confidence output is index 2

    y_pred = np.concatenate(seg_preds, axis=0)
    confidence_scores = np.concatenate(conf_preds, axis=0) if has_confidence else None

    y_p_logits = y_pred[:, 0]
    y_m_logits = y_pred[:, 1]

    y_p_raw = np.argmax(y_p_logits, axis=-1).astype(np.int32)
    y_m_raw = np.argmax(y_m_logits, axis=-1).astype(np.int32)

    y_p_true = y_test[:, 0, :].astype(np.int32)
    y_m_true = y_test[:, 1, :].astype(np.int32)

    raw_p_res = y_p_true - y_p_raw
    raw_m_res = y_m_true - y_m_raw

    print("\n--- Raw Residuals (Before Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(raw_p_res[:, det])), np.std(np.abs(raw_p_res[:, det]))
        m_m, s_m = np.mean(np.abs(raw_m_res[:, det])), np.std(np.abs(raw_m_res[:, det]))
        print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = np.where(mask)[0] + 1
    plot_residuals(
        dets_used, raw_p_res[:, mask], raw_m_res[:, mask], args.model_path, "raw"
    )

    ref_p, ref_m = refine_hit_arrays(y_p_raw, y_m_raw, det_test, elem_test)
    ref_p_res = y_p_true - ref_p
    ref_m_res = y_m_true - ref_m

    print("\n--- Refined Residuals (After Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(ref_p_res[:, det])), np.std(np.abs(ref_p_res[:, det]))
        m_m, s_m = np.mean(np.abs(ref_m_res[:, det])), np.std(np.abs(ref_m_res[:, det]))
        print(f"{det + 1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = np.where(mask)[0] + 1
    plot_residuals(
        dets_used, ref_p_res[:, mask], ref_m_res[:, mask], args.model_path, "refined"
    )

    # Calculate accuracy and chi-squared prior to refinement
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

    # The accuracy/within-2 numbers above are computed over all 62 positions
    # unconditionally, including layers where this specific muon never fired
    # a hit (most of the 62 -- 12 stations are unused entirely, and no real
    # trajectory hits every layer). That's free credit from trivial
    # zero-vs-zero agreement, the same inflation eval_multi_track.py's
    # Hungarian-matched accuracy had before being fixed to mask to non-empty
    # true positions. Reporting the masked version here too so single-track
    # and multi-track numbers are actually comparable.
    hit_mask_p = y_p_true != 0
    hit_mask_m = y_m_true != 0
    raw_res_p_hits = raw_p_res[hit_mask_p]
    raw_res_m_hits = raw_m_res[hit_mask_m]
    pred_hit_p = y_p_raw != 0
    pred_hit_m = y_m_raw != 0
    precision_p = np.mean(y_p_true[pred_hit_p] != 0) if np.any(pred_hit_p) else float("nan")
    precision_m = np.mean(y_m_true[pred_hit_m] != 0) if np.any(pred_hit_m) else float("nan")
    print(f"\n--- Raw, masked to non-empty true positions only ({hit_mask_p.sum()} μ+ / {hit_mask_m.sum()} μ- positions) ---")
    print(f"Raw μ+/μ- accuracy   : {np.mean(np.abs(raw_res_p_hits) == 0):.4f} / {np.mean(np.abs(raw_res_m_hits) == 0):.4f}")
    print(f"Raw μ+/μ- within-2   : {np.mean(np.abs(raw_res_p_hits) <= 2):.4f} / {np.mean(np.abs(raw_res_m_hits) <= 2):.4f}")
    print(f"Raw μ+/μ- mean resid : {np.mean(np.abs(raw_res_p_hits)):.3f} / {np.mean(np.abs(raw_res_m_hits)):.3f} ch")
    print(f"Raw μ+/μ- detection precision: {precision_p:.4f} / {precision_m:.4f}")

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

    ref_res_p_hits = ref_p_res[hit_mask_p]
    ref_res_m_hits = ref_m_res[hit_mask_m]
    print(f"\n--- Refined, masked to non-empty true positions only ---")
    print(f"Refined μ+/μ- accuracy   : {np.mean(np.abs(ref_res_p_hits) == 0):.4f} / {np.mean(np.abs(ref_res_m_hits) == 0):.4f}")
    print(f"Refined μ+/μ- within-2   : {np.mean(np.abs(ref_res_p_hits) <= 2):.4f} / {np.mean(np.abs(ref_res_m_hits) <= 2):.4f}")
    print(f"Refined μ+/μ- mean resid : {np.mean(np.abs(ref_res_p_hits)):.3f} / {np.mean(np.abs(ref_res_m_hits)):.3f} ch")

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

    # ---- Confidence head evaluation ----
    if confidence_scores is not None:
        _evaluate_confidence(
            confidence_scores,
            y_p_raw,
            y_m_raw,
            y_p_true,
            y_m_true,
            args.model_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models."
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
    args = parser.parse_args()

    print(f"\nResults for {args.model_path}...")
    evaluate_model(args)
