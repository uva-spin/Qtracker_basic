import ROOT
import argparse
import numpy as np
from array import array
import time
import os
import random

"""
This version of gen_training creates events with EXACTLY n_pairs ground-truth
dimuon pairs (fixed, not random), plus random extra unpaired single-muon noise
tracks within the remaining budget.

--pairsmup A : Maximum number of mu+ tracks per event (must be >= n_pairs)
--pairsmum B : Maximum number of mu- tracks per event (must be >= n_pairs)
--n_pairs  N : Fixed number of ground-truth dimuon pairs per event
"""

NUM_LAYERS = 62


def _get_value(x):
    if hasattr(x, "__getitem__"):
        return x[0]
    return x


def combine_files(file1, file2, output_file, pairsmup, pairsmum, n_pairs):
    if os.path.exists(output_file):
        os.remove(output_file)

    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")

    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)

    output_tree = ROOT.TTree("tree", "Tree with aligned dimuon pairs")
    output_tree.SetAutoSave(0)

    # ---------------- Event-level branches ----------------
    event_id = array("i", [0])
    nPairs = array("i", [0])
    nExtraMup = array("i", [0])
    nExtraMum = array("i", [0])

    output_tree.Branch("eventID", event_id, "eventID/I")
    output_tree.Branch("nPairs", nPairs, "nPairs/I")
    output_tree.Branch("nExtraMup", nExtraMup, "nExtraMup/I")
    output_tree.Branch("nExtraMum", nExtraMum, "nExtraMum/I")

    # ---------------- Flat hit storage ----------------
    mu_id = ROOT.std.vector("int")()
    element_id = ROOT.std.vector("int")()
    detector_id = ROOT.std.vector("int")()
    drift_distance = ROOT.std.vector("double")()
    tdc_time = ROOT.std.vector("double")()
    hit_id = ROOT.std.vector("int")()
    hit_track_id = ROOT.std.vector("int")()
    gprocess_id = ROOT.std.vector("int")()

    gcharge = ROOT.std.vector("int")()
    gtrack_id = ROOT.std.vector("int")()
    gpx = ROOT.std.vector("double")()
    gpy = ROOT.std.vector("double")()
    gpz = ROOT.std.vector("double")()
    gvx = ROOT.std.vector("double")()
    gvy = ROOT.std.vector("double")()
    gvz = ROOT.std.vector("double")()

    output_tree.Branch("muID", mu_id)
    output_tree.Branch("elementID", element_id)
    output_tree.Branch("detectorID", detector_id)
    output_tree.Branch("driftDistance", drift_distance)
    output_tree.Branch("tdcTime", tdc_time)
    output_tree.Branch("hitID", hit_id)
    output_tree.Branch("hitTrackID", hit_track_id)
    output_tree.Branch("gProcessID", gprocess_id)

    output_tree.Branch("gCharge", gcharge)
    output_tree.Branch("gTrackID", gtrack_id)
    output_tree.Branch("gpx", gpx)
    output_tree.Branch("gpy", gpy)
    output_tree.Branch("gpz", gpz)
    output_tree.Branch("gvx", gvx)
    output_tree.Branch("gvy", gvy)
    output_tree.Branch("gvz", gvz)

    # ---------------- Structured paired arrays ----------------
    hitarray_mup = np.zeros((n_pairs, NUM_LAYERS), dtype=np.int32)
    hitarray_mum = np.zeros((n_pairs, NUM_LAYERS), dtype=np.int32)

    output_tree.Branch(
        "HitArray_mup", hitarray_mup, f"HitArray_mup[{n_pairs}][{NUM_LAYERS}]/I"
    )
    output_tree.Branch(
        "HitArray_mum", hitarray_mum, f"HitArray_mum[{n_pairs}][{NUM_LAYERS}]/I"
    )

    n1 = tree1.GetEntries()
    n2 = tree2.GetEntries()

    idx_mup = 0
    idx_mum = 0
    ev = 0

    while idx_mup < n1 and idx_mum < n2:
        # Fixed n_pairs pairs, random extra singles within budget
        current_pairs = n_pairs
        extra_mup = random.randint(0, pairsmup - n_pairs)
        extra_mum = random.randint(0, pairsmum - n_pairs)

        if idx_mup + current_pairs + extra_mup > n1:
            break
        if idx_mum + current_pairs + extra_mum > n2:
            break

        # Reset event
        event_id[0] = ev
        nPairs[0] = current_pairs
        nExtraMup[0] = extra_mup
        nExtraMum[0] = extra_mum

        mu_id.clear()
        element_id.clear()
        detector_id.clear()
        drift_distance.clear()
        tdc_time.clear()
        hit_id.clear()
        hit_track_id.clear()
        gprocess_id.clear()
        gcharge.clear()
        gtrack_id.clear()
        gpx.clear()
        gpy.clear()
        gpz.clear()
        gvx.clear()
        gvy.clear()
        gvz.clear()

        hitarray_mup.fill(0)
        hitarray_mum.fill(0)

        track_counter = 0

        # -------- Fill paired tracks --------
        for k in range(current_pairs):
            tree1.GetEntry(idx_mup)
            tree2.GetEntry(idx_mum)

            # Fill structured arrays
            for elem, det in zip(tree1.elementID, tree1.detectorID):
                if 1 <= det <= NUM_LAYERS:
                    hitarray_mup[k, det - 1] = elem

            for elem, det in zip(tree2.elementID, tree2.detectorID):
                if 1 <= det <= NUM_LAYERS:
                    hitarray_mum[k, det - 1] = elem

            # Append flat hits from both
            for source in [tree1, tree2]:
                for elem, det, drift, tdc, hid, trk, proc in zip(
                    source.elementID,
                    source.detectorID,
                    source.driftDistance,
                    source.tdcTime,
                    source.hitID,
                    source.hitTrackID,
                    source.gProcessID,
                ):
                    element_id.push_back(elem)
                    detector_id.push_back(det)
                    drift_distance.push_back(drift)
                    tdc_time.push_back(tdc)
                    hit_id.push_back(hid)
                    hit_track_id.push_back(trk)
                    gprocess_id.push_back(proc)

                mu_id.push_back(track_counter + 1)
                gcharge.push_back(_get_value(source.gCharge))
                gtrack_id.push_back(_get_value(source.gTrackID))
                gpx.push_back(_get_value(source.gpx))
                gpy.push_back(_get_value(source.gpy))
                gpz.push_back(_get_value(source.gpz))
                gvx.push_back(_get_value(source.gvx))
                gvy.push_back(_get_value(source.gvy))
                gvz.push_back(_get_value(source.gvz))

                track_counter += 1

            idx_mup += 1
            idx_mum += 1

        # -------- Extra mu+ --------
        for _ in range(extra_mup):
            tree1.GetEntry(idx_mup)
            for elem, det, drift, tdc, hid, trk, proc in zip(
                tree1.elementID,
                tree1.detectorID,
                tree1.driftDistance,
                tree1.tdcTime,
                tree1.hitID,
                tree1.hitTrackID,
                tree1.gProcessID,
            ):
                element_id.push_back(elem)
                detector_id.push_back(det)
                drift_distance.push_back(drift)
                tdc_time.push_back(tdc)
                hit_id.push_back(hid)
                hit_track_id.push_back(trk)
                gprocess_id.push_back(proc)

            idx_mup += 1

        # -------- Extra mu− --------
        for _ in range(extra_mum):
            tree2.GetEntry(idx_mum)
            for elem, det, drift, tdc, hid, trk, proc in zip(
                tree2.elementID,
                tree2.detectorID,
                tree2.driftDistance,
                tree2.tdcTime,
                tree2.hitID,
                tree2.hitTrackID,
                tree2.gProcessID,
            ):
                element_id.push_back(elem)
                detector_id.push_back(det)
                drift_distance.push_back(drift)
                tdc_time.push_back(tdc)
                hit_id.push_back(hid)
                hit_track_id.push_back(trk)
                gprocess_id.push_back(proc)

            idx_mum += 1

        output_tree.Fill()
        ev += 1

    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f1.Close()
    f2.Close()

    fout = ROOT.TFile.Open(output_file, "READ")
    out_tree = fout.Get("tree")
    print(f"Events in output tree after writing: {out_tree.GetEntries()}")
    print(
        "Trees in output file:",
        [
            key.GetName()
            for key in fout.GetListOfKeys()
            if key.GetClassName() == "TTree"
        ],
    )
    cycles = [
        key.GetCycle() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"
    ]
    print("Tree cycles in output file:", cycles)
    fout.Close()


def add_hit_array(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    f_in = ROOT.TFile.Open(input_file, "READ")
    tree = f_in.Get("tree")

    print(f"Entries in input file {input_file}: {tree.GetEntries()}")
    print(
        "Trees in input file:",
        [
            key.GetName()
            for key in f_in.GetListOfKeys()
            if key.GetClassName() == "TTree"
        ],
    )

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)

    output_tree = tree.CloneTree(0)
    output_tree.SetAutoSave(0)

    hit_array = np.zeros((NUM_LAYERS, 2), dtype=np.float64)
    output_tree.Branch("HitArray", hit_array, f"HitArray[{NUM_LAYERS}][2]/D")

    fill_count = 0
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        hit_array.fill(0)

        for elem, det, drift in zip(
            tree.elementID, tree.detectorID, tree.driftDistance
        ):
            if 1 <= det <= NUM_LAYERS:
                hit_array[det - 1, 0] = elem
                hit_array[det - 1, 1] = drift if elem != 0 else 0.0

        output_tree.Fill()
        fill_count += 1

    print(f"Fill() called {fill_count} times for {output_file}")
    print(f"Events in output tree before writing: {output_tree.GetEntries()}")

    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f_in.Close()

    fout = ROOT.TFile.Open(output_file, "READ")
    out_tree = fout.Get("tree")
    print(f"Events in output tree after writing: {out_tree.GetEntries()}")
    print(
        "Trees in output file:",
        [
            key.GetName()
            for key in fout.GetListOfKeys()
            if key.GetClassName() == "TTree"
        ],
    )
    cycles = [
        key.GetCycle() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"
    ]
    print("Tree cycles in output file:", cycles)
    fout.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine ROOT trees with a fixed number of dimuon pairs per event."
    )
    parser.add_argument("file1", type=str, help="First ROOT file (mu+)")
    parser.add_argument("file2", type=str, help="Second ROOT file (mu-)")
    parser.add_argument(
        "--output", type=str, default="finder_training.root", help="Output file"
    )
    parser.add_argument(
        "--pairsmup", type=int, default=None, help="Max number of mu+ tracks per event"
    )
    parser.add_argument(
        "--pairsmum", type=int, default=None, help="Max number of mu- tracks per event"
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        required=True,
        help="Fixed number of ground-truth dimuon pairs per event",
    )

    args = parser.parse_args()

    if args.n_pairs < 1:
        raise ValueError("--n_pairs must be >= 1")

    if args.pairsmup is None and args.pairsmum is None:
        args.pairsmup = args.n_pairs
        args.pairsmum = args.n_pairs
    elif args.pairsmup is None:
        args.pairsmup = args.pairsmum
    elif args.pairsmum is None:
        args.pairsmum = args.pairsmup

    if args.pairsmup < args.n_pairs:
        raise ValueError("--pairsmup must be >= --n_pairs")
    if args.pairsmum < args.n_pairs:
        raise ValueError("--pairsmum must be >= --n_pairs")

    file1_array_output = "momentum_training-1.root"
    file2_array_output = "momentum_training-2.root"
    for file_name in [args.output, file1_array_output, file2_array_output]:
        if os.path.exists(file_name):
            os.remove(file_name)

    print(f"ROOT version: {ROOT.gROOT.GetVersion()}")
    start_time = time.time()
    combine_files(
        args.file1,
        args.file2,
        args.output,
        args.pairsmup,
        args.pairsmum,
        args.n_pairs,
    )
    end_time = time.time()

    start_time = time.time()
    add_hit_array(args.file1, file1_array_output)
    add_hit_array(args.file2, file2_array_output)
    end_time = time.time()

    print(f"Generated: {args.output}, {file1_array_output}, {file2_array_output}")
