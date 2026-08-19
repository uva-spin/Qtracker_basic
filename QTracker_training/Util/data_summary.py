"""
Summarize the multi-track training/validation ROOT files.
Run on Rivanna where data lives:

    python3 Util/data_summary.py \
        /project/ptgroup/spinquest/Anvesh/data/multi_track/processed_files/
"""

import argparse
import os
import numpy as np

try:
    import uproot
    _BACKEND = "uproot"
except ImportError:
    import ROOT
    _BACKEND = "ROOT"


def summarize_file_uproot(path: str, max_pairs: int = 3) -> dict:
    with uproot.open(path) as f:
        tree = f["tree"]
        n_events = tree.num_entries
        mup = tree["HitArray_mup"].array(library="np")   # (N, max_pairs*62) flat
        mum = tree["HitArray_mum"].array(library="np")
        mup = mup[:, :max_pairs * 62].reshape(n_events, max_pairs, 62)
        mum = mum[:, :max_pairs * 62].reshape(n_events, max_pairs, 62)
    return _compute_stats(n_events, mup, mum, max_pairs)


def summarize_file_root(path: str, max_pairs: int = 3) -> dict:
    import ROOT as R
    f = R.TFile.Open(path, "READ")
    tree = f.Get("tree")
    n_events = tree.GetEntries()
    mup_list, mum_list = [], []
    for ev in tree:
        mup_list.append(list(ev.HitArray_mup)[:max_pairs * 62])
        mum_list.append(list(ev.HitArray_mum)[:max_pairs * 62])
    f.Close()
    mup = np.array(mup_list).reshape(n_events, max_pairs, 62)
    mum = np.array(mum_list).reshape(n_events, max_pairs, 62)
    return _compute_stats(n_events, mup, mum, max_pairs)


def _compute_stats(n_events: int, mup: np.ndarray, mum: np.ndarray, max_pairs: int) -> dict:
    pair_counts = np.zeros(max_pairs + 1, dtype=int)
    for p in range(max_pairs):
        has_pair = np.any(mup[:, p, :] != 0, axis=1) | np.any(mum[:, p, :] != 0, axis=1)
        pair_counts[p + 1] = int(np.sum(has_pair))  # cumulative (slots filled sequentially)

    # actual pair count per event: count filled slots
    filled = np.zeros(n_events, dtype=int)
    for p in range(max_pairs):
        has = np.any(mup[:, p, :] != 0, axis=1) | np.any(mum[:, p, :] != 0, axis=1)
        filled += has.astype(int)
    dist = {k: int(np.sum(filled == k)) for k in range(max_pairs + 1)}

    return {"n_events": n_events, "pair_distribution": dist}


def summarize(data_dir: str, max_pairs: int = 3) -> None:
    files = {
        "train_low":  "mc_events_train_low.root",
        "train_med":  "mc_events_train_med.root",
        "train_high": "mc_events_train_high.root",
        "val":        "mc_events_val.root",
    }

    fn = summarize_file_uproot if _BACKEND == "uproot" else summarize_file_root
    print(f"Backend: {_BACKEND}\n")
    print(f"{'File':<18} {'N events':>10}  {'0 pairs':>8}  {'1 pair':>8}  {'2 pairs':>8}  {'3 pairs':>8}")
    print("-" * 68)

    total = 0
    for label, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  {label:<16}  NOT FOUND: {path}")
            continue
        stats = fn(path, max_pairs)
        n = stats["n_events"]
        dist = stats["pair_distribution"]
        total += n
        row = f"  {label:<16}  {n:>10,}"
        for k in range(max_pairs + 1):
            row += f"  {dist.get(k, 0):>8,}"
        print(row)

    print("-" * 68)
    print(f"  {'TOTAL':<16}  {total:>10,}")
    print(f"\nmax_pairs={max_pairs} (slots filled sequentially; pair_distribution shows events per slot count)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize multi-track ROOT data files.")
    parser.add_argument("data_dir", type=str, help="Directory containing mc_events_*.root files.")
    parser.add_argument("--max_pairs", type=int, default=3)
    args = parser.parse_args()
    summarize(args.data_dir, args.max_pairs)
