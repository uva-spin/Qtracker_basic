"""
Generate residual training data for fine-tuning the autoregressive track finder.

For each multi-track event with K pairs:
- Iteration 0: original hit matrix → GT pair 0
- Iteration 1: hit matrix minus pair 0's hits → GT pair 1
- ...
- Iteration K-1: hit matrix minus pairs 0..K-2 → GT pair K-1

Output: single-track format ROOT file where each row is one (residual_matrix, single_pair) sample.
"""

import ROOT
import numpy as np
from array import array
import argparse
import os

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def generate_residual_data(input_file, output_file, max_pairs):
    f_in = ROOT.TFile.Open(input_file, "READ")
    tree = f_in.Get("tree")
    if not tree:
        raise RuntimeError(f"Tree not found in {input_file}")

    if os.path.exists(output_file):
        os.remove(output_file)

    f_out = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    f_out.SetCompressionLevel(5)
    out_tree = ROOT.TTree("tree", "Residual training data for autoregressive finder")
    out_tree.SetAutoSave(0)

    # Output branches: single-track format
    eventID = array("i", [0])
    iterationID = array("i", [0])

    element_id = ROOT.std.vector("int")()
    detector_id = ROOT.std.vector("int")()
    element_id_clean = ROOT.std.vector("int")()
    detector_id_clean = ROOT.std.vector("int")()
    drift_distance = ROOT.std.vector("double")()
    tdc_time = ROOT.std.vector("double")()

    HitArray_mup = np.zeros(NUM_DETECTORS, dtype=np.int32)
    HitArray_mum = np.zeros(NUM_DETECTORS, dtype=np.int32)

    out_tree.Branch("eventID", eventID, "eventID/I")
    out_tree.Branch("iterationID", iterationID, "iterationID/I")
    out_tree.Branch("elementID", element_id)
    out_tree.Branch("detectorID", detector_id)
    out_tree.Branch("elementIDClean", element_id_clean)
    out_tree.Branch("detectorIDClean", detector_id_clean)
    out_tree.Branch("driftDistance", drift_distance)
    out_tree.Branch("tdcTime", tdc_time)
    out_tree.Branch("HitArray_mup", HitArray_mup, f"HitArray_mup[{NUM_DETECTORS}]/I")
    out_tree.Branch("HitArray_mum", HitArray_mum, f"HitArray_mum[{NUM_DETECTORS}]/I")

    # Bind input multi-track hit arrays
    leaf_mup = tree.GetLeaf("HitArray_mup")
    total_len = leaf_mup.GetLen() if leaf_mup else 0
    input_max_pairs = total_len // NUM_DETECTORS if total_len > 0 else max_pairs

    input_mup = np.zeros((input_max_pairs, NUM_DETECTORS), dtype=np.int32)
    input_mum = np.zeros((input_max_pairs, NUM_DETECTORS), dtype=np.int32)
    tree.SetBranchAddress("HitArray_mup", input_mup)
    tree.SetBranchAddress("HitArray_mum", input_mum)

    fills = 0
    for ev in range(tree.GetEntries()):
        tree.GetEntry(ev)

        n_active = int(tree.nPairs) if hasattr(tree, "nPairs") else input_max_pairs

        # Build the full hit set for this event
        all_elem = list(tree.elementID)
        all_det = list(tree.detectorID)
        all_drift = list(tree.driftDistance)
        all_tdc = list(tree.tdcTime)

        all_elem_clean = (
            list(tree.elementIDClean) if hasattr(tree, "elementIDClean") else all_elem
        )
        all_det_clean = (
            list(tree.detectorIDClean) if hasattr(tree, "detectorIDClean") else all_det
        )

        # Track which hits have been "consumed" by previous iterations
        removed_positions = set()  # (det, elem) tuples

        for it in range(n_active):
            # Check if this pair is actually active (nonzero)
            if np.all(input_mup[it] == 0) and np.all(input_mum[it] == 0):
                continue

            eventID[0] = ev
            iterationID[0] = it

            # Write residual hit vectors (excluding removed positions)
            element_id.clear()
            detector_id.clear()
            element_id_clean.clear()
            detector_id_clean.clear()
            drift_distance.clear()
            tdc_time.clear()

            for elem, det, drift, tdc in zip(all_elem, all_det, all_drift, all_tdc):
                if (int(det), int(elem)) not in removed_positions:
                    element_id.push_back(int(elem))
                    detector_id.push_back(int(det))
                    drift_distance.push_back(float(drift))
                    tdc_time.push_back(float(tdc))

            for elem, det in zip(all_elem_clean, all_det_clean):
                if (int(det), int(elem)) not in removed_positions:
                    element_id_clean.push_back(int(elem))
                    detector_id_clean.push_back(int(det))

            # GT for this iteration: the current pair
            for d in range(NUM_DETECTORS):
                HitArray_mup[d] = input_mup[it, d]
                HitArray_mum[d] = input_mum[it, d]

            out_tree.Fill()
            fills += 1

            # Mark this pair's hits as removed for next iteration
            for d in range(NUM_DETECTORS):
                if input_mup[it, d] > 0:
                    removed_positions.add((d + 1, int(input_mup[it, d])))
                if input_mum[it, d] > 0:
                    removed_positions.add((d + 1, int(input_mum[it, d])))

    print(
        f"Generated {fills} residual training samples from {tree.GetEntries()} events."
    )

    out_tree.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate residual training data for autoregressive track finder."
    )
    parser.add_argument("input_file", type=str, help="Multi-track mc_events ROOT file.")
    parser.add_argument(
        "--output",
        type=str,
        default="residual_training.root",
        help="Output single-track-format ROOT file.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="Max pairs in the input file.",
    )
    args = parser.parse_args()
    generate_residual_data(args.input_file, args.output, args.max_pairs)
