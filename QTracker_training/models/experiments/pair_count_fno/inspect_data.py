"""
Diagnostic: inspect the multi-track curriculum ROOT files directly for
structural differences between low/med/high complexity phases and the
validation set, beyond just injected background-track count.

Motivation: the Axial-FNO pair-count classifier -- and, per project
history, the joint MultiTrackFinder model -- shows val_loss diverging hard
the moment training data switches to the "high" complexity phase (34-50
background tracks/event). This checks whether that's a genuine train/val
distribution mismatch or a data-generation inconsistency:
  - Does the nPairs truth branch (set in gen_training_random.py, carried
    through by messy_gen.py) agree with the pair-count label derived from
    HitArray occupancy (what load_data_pair_count actually uses)?
  - What fraction of events per file have out-of-range HitArray values,
    which load_data_denoise silently drops via `continue`?
  - Does track/hit density actually escalate low -> med -> high as
    documented, and does the validation file really span the full range,
    or is it concentrated at low/med complexity (which would make training
    purely on "high" data a genuine domain shift, not a bug)?

Usage (inside the apptainer container on Rivanna):
    python3 inspect_data.py \
        /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
        /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
        /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
        /mnt/data/data/multi_track/processed_files/mc_events_val.root

Add --limit N to scan only the first N events per file for a quick pass.
"""

import argparse
import os

import numpy as np
import ROOT  # noqa: F401

NUM_LAYERS = 62
NUM_ELEMENT_IDS = 201


def _summarize(name: str, arr: np.ndarray) -> str:
    return f"mean={arr.mean():.1f}, std={arr.std():.1f}, min={arr.min()}, max={arr.max()}"


def inspect_file(path: str, max_pairs: int, limit: int) -> None:
    f = ROOT.TFile.Open(path, "READ")
    tree = f.Get("tree")
    if not tree:
        print(f"  Could not open tree in {path}")
        return

    n_events = tree.GetEntries()
    n_scan = min(n_events, limit) if limit else n_events

    nPairs_truth = []
    nPairs_derived = []
    mismatches = 0
    track_counts = []
    hit_counts = []
    n_range_invalid = 0

    for i, event in enumerate(tree):
        if i >= n_scan:
            break

        truth_n = int(event.nPairs)
        nPairs_truth.append(truth_n)

        mu_plus = np.array(list(event.HitArray_mup), dtype=np.int64)
        mu_minus = np.array(list(event.HitArray_mum), dtype=np.int64)
        mu_plus = mu_plus[: max_pairs * NUM_LAYERS].reshape(max_pairs, NUM_LAYERS)
        mu_minus = mu_minus[: max_pairs * NUM_LAYERS].reshape(max_pairs, NUM_LAYERS)

        out_of_range = (
            np.any(mu_plus < 0) or np.any(mu_plus >= NUM_ELEMENT_IDS)
            or np.any(mu_minus < 0) or np.any(mu_minus >= NUM_ELEMENT_IDS)
        )
        if out_of_range:
            n_range_invalid += 1

        occupied = np.any(mu_plus != 0, axis=1) | np.any(mu_minus != 0, axis=1)
        derived_n = int(occupied.sum())
        nPairs_derived.append(derived_n)
        if derived_n != truth_n:
            mismatches += 1

        track_counts.append(len(event.gTrackID))
        hit_counts.append(len(event.elementID))

    nPairs_truth = np.array(nPairs_truth)
    nPairs_derived = np.array(nPairs_derived)
    track_counts = np.array(track_counts)
    hit_counts = np.array(hit_counts)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Total events in file: {n_events} (scanned {n_scan})")
    truth_vals, truth_counts = np.unique(nPairs_truth, return_counts=True)
    derived_vals, derived_counts = np.unique(nPairs_derived, return_counts=True)
    print(f"nPairs (truth branch) distribution:      {dict(zip(truth_vals.tolist(), truth_counts.tolist()))}")
    print(f"nPairs (derived from HitArray) distribution: {dict(zip(derived_vals.tolist(), derived_counts.tolist()))}")
    print(f"Mismatches truth vs derived: {mismatches} ({100 * mismatches / n_scan:.2f}%)")
    print(
        f"HitArray range-invalid events (silently dropped by load_data_denoise): "
        f"{n_range_invalid} ({100 * n_range_invalid / n_scan:.2f}%)"
    )
    print(f"Tracks/event (len(gTrackID)): {_summarize('tracks', track_counts)}")
    print(f"Hits/event (len(elementID)): {_summarize('hits', hit_counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_files", nargs="+", help="ROOT files to inspect, in any order.")
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Scan only the first N events per file (0 = all).")
    args = parser.parse_args()
    for path in args.root_files:
        inspect_file(path, args.max_pairs, args.limit)
