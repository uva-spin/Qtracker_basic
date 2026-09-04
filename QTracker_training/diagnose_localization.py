# ruff: noqa: E402
"""
Diagnoses whether the segmenter's poor localization accuracy (~9-13% exact,
per eval_multi_track.py's corrected, non-empty-masked Hungarian metrics) is
a genuine architecture/learning problem, or an argmax-decoding problem that
peak detection / soft-argmax could fix cheaply without retraining.

For every (event, pair, charge, detector) position where a real hit exists
and the model's argmax prediction is wrong, checks:
  - the rank of the true elementID within that position's softmax
    distribution (is it near the top, just not #1?)
  - the softmax probability mass in a window around the true position vs.
    around the argmax position (is mass actually concentrated near truth,
    with argmax just landing on a narrow spurious spike elsewhere?)
  - whether a probability-weighted "soft-argmax" centroid lands closer to
    truth than the raw argmax

If the true position consistently has high rank / most of the nearby mass,
that points to a cheap peak-detection/decoding fix (Solution 1's "peak
detection" half). If the true position is nowhere near where the softmax
mass concentrates, argmax isn't the problem -- the U-Net genuinely isn't
learning to localize, and only an architecture change (asymmetric kernels)
would help.

Usage:
    python3 diagnose_localization.py <model.keras> <val_root_file> [--max_pairs 3] [--num_events 5000]
"""

import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ROOT  # noqa: F401
import numpy as np
import tensorflow as tf

from models import data_loader
from models.layers import AxialAttention

NUM_ELEMENT_IDS = 201


def analyze(model_path: str, root_file: str, max_pairs: int, num_events, window: int) -> None:
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(
        root_file, multi_track=True, max_pairs=max_pairs
    )
    if X_test is None:
        raise RuntimeError(f"Could not load {root_file}")
    if num_events:
        X_test = X_test[:num_events]
        y_muPlus_test = y_muPlus_test[:num_events]
        y_muMinus_test = y_muMinus_test[:num_events]

    y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=2)  # (N, max_pairs, 2, 62)

    custom_objects = {"AxialAttention": AxialAttention}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)

    preds = []
    chunk = 128
    for i in range(0, len(X_test), chunk):
        X_chunk = tf.cast(X_test[i:i + chunk], tf.float32)
        y_chunk = model.predict(X_chunk, verbose=0)
        preds.append(y_chunk[1])
    y_pred = np.concatenate(preds, axis=0)  # (N, max_pairs, 2, 62, 201)
    y_pred_argmax = np.argmax(y_pred, axis=-1)

    hit_mask = y_test != 0
    true_ids = y_test[hit_mask].astype(np.int32)
    pred_ids = y_pred_argmax[hit_mask]
    probs = y_pred[hit_mask]  # (M, 201)

    wrong_mask = pred_ids != true_ids
    n_total, n_wrong = len(true_ids), int(wrong_mask.sum())
    print(f"Positions with a real hit: {n_total}")
    print(f"Argmax-wrong positions: {n_wrong} ({100 * n_wrong / n_total:.1f}%)")

    probs_wrong = probs[wrong_mask]
    true_wrong = true_ids[wrong_mask]
    pred_wrong = pred_ids[wrong_mask]

    true_prob = np.take_along_axis(probs_wrong, true_wrong[:, None], axis=1)[:, 0]
    ranks = (probs_wrong > true_prob[:, None]).sum(axis=1)
    print("\nOf argmax-wrong positions, rank of the true elementID in the softmax "
          "(rank 0 would mean it was the argmax -- always >=1 here by construction):")
    for k in (3, 5, 10, 20):
        print(f"  true ID within top-{k}: {100 * np.mean(ranks < k):.1f}%")
    print(f"  median rank: {np.median(ranks):.0f} / {NUM_ELEMENT_IDS}")

    idx = np.arange(NUM_ELEMENT_IDS)

    def window_mass(centers: np.ndarray) -> np.ndarray:
        lo = np.clip(centers - window, 0, NUM_ELEMENT_IDS)
        hi = np.clip(centers + window + 1, 0, NUM_ELEMENT_IDS)
        return np.array([probs_wrong[i, l:h].sum() for i, (l, h) in enumerate(zip(lo, hi))])

    mass_near_true = window_mass(true_wrong)
    mass_near_pred = window_mass(pred_wrong)
    print(f"\nMean probability mass within +/-{window} channels of TRUE position:   {mass_near_true.mean():.4f}")
    print(f"Mean probability mass within +/-{window} channels of ARGMAX position: {mass_near_pred.mean():.4f}")

    soft_pred = probs_wrong @ idx
    res_argmax = np.abs(true_wrong.astype(float) - pred_wrong.astype(float))
    res_soft = np.abs(true_wrong.astype(float) - soft_pred)
    print(f"\nMean residual, raw argmax:  {res_argmax.mean():.2f} channels")
    print(f"Mean residual, soft-argmax: {res_soft.mean():.2f} channels")
    print(f"Soft-argmax closer than raw argmax on {100 * np.mean(res_soft < res_argmax):.1f}% of wrong positions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=str)
    parser.add_argument("root_file", type=str)
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--num_events", type=int, default=5000, help="Subsample for speed (0 = all).")
    parser.add_argument("--window", type=int, default=10, help="Channel window for probability-mass check.")
    args = parser.parse_args()

    analyze(args.model_path, args.root_file, args.max_pairs, args.num_events or None, args.window)
