"""
root_to_numpy.py — Convert multi-track ROOT files to compressed numpy archives (.npz)
for use in ROOT-free environments (Google Colab, local Mac, etc.)

Usage (run inside the Apptainer container on Rivanna):
    apptainer exec --bind /mnt/data /project/ptgroup/spinquest/David/TfRootBuild.sif \
        python3 /mnt/code/data/root_to_numpy.py \
        --data_dir /mnt/data/data/multi_track/processed_files \
        --out_dir  /mnt/data/data/multi_track/numpy_files \
        --max_pairs 5

Or as a SLURM job (no GPU needed, just CPU + ROOT):
    sbatch scripts/convert_numpy.slurm
"""

import argparse
import os
import sys

import numpy as np
import ROOT  # noqa: F401


def convert_file(root_path: str, out_path: str, max_pairs: int) -> None:
    num_detectors = 62
    num_elementIDs = 201

    f = ROOT.TFile.Open(root_path, "READ")
    tree = f.Get("tree")
    if not tree:
        print(f"ERROR: no tree found in {root_path}", flush=True)
        sys.exit(1)

    n_events = tree.GetEntries()
    print(f"  {os.path.basename(root_path)}: {n_events} events", flush=True)

    X = np.zeros((n_events, num_detectors, num_elementIDs, 1), dtype=np.float32)
    X_clean = np.zeros((n_events, num_detectors, num_elementIDs, 1), dtype=np.float32)
    y_mup = np.zeros((n_events, max_pairs, num_detectors), dtype=np.int8)
    y_mum = np.zeros((n_events, max_pairs, num_detectors), dtype=np.int8)

    for i, event in enumerate(tree):
        if i % 10000 == 0:
            print(f"    {i}/{n_events}", flush=True)

        for det_id, elem_id in zip(event.detectorID, event.elementID):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                X[i, det_id, elem_id, 0] = 1.0

        for det_id, elem_id in zip(event.detectorIDClean, event.elementIDClean):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                X_clean[i, det_id, elem_id, 0] = 1.0

        mup = np.array(list(event.HitArray_mup), dtype=np.int8).reshape(max_pairs, num_detectors)
        mum = np.array(list(event.HitArray_mum), dtype=np.int8).reshape(max_pairs, num_detectors)
        y_mup[i] = np.clip(mup, 0, 1)
        y_mum[i] = np.clip(mum, 0, 1)

    f.Close()

    np.savez_compressed(
        out_path,
        X=X,
        X_clean=X_clean,
        y_mup=y_mup,
        y_mum=y_mum,
    )
    size_mb = os.path.getsize(out_path + ".npz") / 1e6
    print(f"  Saved {out_path}.npz ({size_mb:.0f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing ROOT files")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    parser.add_argument("--max_pairs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = {
        "mc_events_train_low": "train_low",
        "mc_events_train_med": "train_med",
        "mc_events_train_high": "train_high",
        "mc_events_val": "val",
    }

    for root_name, out_name in files.items():
        root_path = os.path.join(args.data_dir, root_name + ".root")
        out_path = os.path.join(args.out_dir, out_name)
        if not os.path.exists(root_path):
            print(f"Skipping {root_path} (not found)", flush=True)
            continue
        print(f"Converting {root_name} ...", flush=True)
        convert_file(root_path, out_path, args.max_pairs)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
