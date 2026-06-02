import ROOT
import argparse
import numpy as np
from array import array
import time
import os


def combine_files(file1, file2, output_file, n_pairs=1):
    if os.path.exists(output_file):
        os.remove(output_file)

    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")

    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    print(f"Entries in {file1}: {tree1.GetEntries()}")
    print(f"Entries in {file2}: {tree2.GetEntries()}")
    print(
        "Trees in file1:",
        [key.GetName() for key in f1.GetListOfKeys() if key.GetClassName() == "TTree"],
    )
    print(
        "Trees in file2:",
        [key.GetName() for key in f2.GetListOfKeys() if key.GetClassName() == "TTree"],
    )

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)

    # Create the output tree
    output_tree = ROOT.TTree("tree", "Tree with combined hits and track information")

    # Disable auto-save to prevent intermediate writes
    output_tree.SetAutoSave(0)  # Disable auto-save (0 or negative value)

    eventID = array("i", [0])
    nPairs = array("i", [0])
    muID = ROOT.std.vector("int")()
    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()
    hitID = ROOT.std.vector("int")()
    hitTrackID = ROOT.std.vector("int")()
    gProcessID = ROOT.std.vector("int")()
    gCharge = ROOT.std.vector("int")()
    gTrackID = ROOT.std.vector("int")()
    gpx = ROOT.std.vector("double")()
    gpy = ROOT.std.vector("double")()
    gpz = ROOT.std.vector("double")()
    gvx = ROOT.std.vector("double")()
    gvy = ROOT.std.vector("double")()
    gvz = ROOT.std.vector("double")()

    # GT hit arrays as 2D: [pair][layer]
    NUM_LAYERS = 62
    HitArray_mup = np.zeros((n_pairs, NUM_LAYERS), dtype=np.int32)
    HitArray_mum = np.zeros((n_pairs, NUM_LAYERS), dtype=np.int32)

    output_tree.Branch("eventID", eventID, "eventID/I")
    output_tree.Branch("nPairs", nPairs, "nPairs/I")
    output_tree.Branch("muID", muID)
    output_tree.Branch("elementID", elementID)
    output_tree.Branch("detectorID", detectorID)
    output_tree.Branch("driftDistance", driftDistance)
    output_tree.Branch("tdcTime", tdcTime)
    output_tree.Branch("hitID", hitID)
    output_tree.Branch("hitTrackID", hitTrackID)
    output_tree.Branch("gProcessID", gProcessID)
    output_tree.Branch("gCharge", gCharge)
    output_tree.Branch("gTrackID", gTrackID)
    output_tree.Branch("gpx", gpx)
    output_tree.Branch("gpy", gpy)
    output_tree.Branch("gpz", gpz)
    output_tree.Branch("gvx", gvx)
    output_tree.Branch("gvy", gvy)
    output_tree.Branch("gvz", gvz)
    # Note: branch names keep the same identifiers; only the leaf shape changes to [nPairs][62]
    output_tree.Branch(
        "HitArray_mup", HitArray_mup, f"HitArray_mup[{n_pairs}][{NUM_LAYERS}]/I"
    )
    output_tree.Branch(
        "HitArray_mum", HitArray_mum, f"HitArray_mum[{n_pairs}][{NUM_LAYERS}]/I"
    )

    def _v(x):
        return x[0] if hasattr(x, "__getitem__") else x

    fills = 0
    n1 = tree1.GetEntries()
    n2 = tree2.GetEntries()
    n_min = min(n1, n2)
    if n_pairs <= 0:
        raise ValueError("n_pairs must be >= 1")
    max_events = (
        n_min // n_pairs
    )  # consume n_pairs entries from each tree per output event

    for ev in range(max_events):
        base = ev * n_pairs

        muID.clear()
        elementID.clear()
        detectorID.clear()
        driftDistance.clear()
        gProcessID.clear()
        tdcTime.clear()
        hitID.clear()
        hitTrackID.clear()
        gpx.clear()
        gpy.clear()
        gpz.clear()
        gvx.clear()
        gvy.clear()
        gvz.clear()
        gCharge.clear()
        gTrackID.clear()
        HitArray_mup.fill(0)
        HitArray_mum.fill(0)

        eventID[0] = ev
        nPairs[0] = n_pairs

        # For each pair k: tree1[base+k] -> mu+, tree2[base+k] -> mu-
        for k in range(n_pairs):
            idx = base + k
            tree1.GetEntry(idx)
            tree2.GetEntry(idx)

            # accumulate raw hits (union of four tracks over all pairs)
            for elem, det, drift, tdc, hit, track, proc in zip(
                tree1.elementID,
                tree1.detectorID,
                tree1.driftDistance,
                tree1.tdcTime,
                tree1.hitID,
                tree1.hitTrackID,
                tree1.gProcessID,
            ):
                elementID.push_back(elem)
                detectorID.push_back(det)
                driftDistance.push_back(drift)
                tdcTime.push_back(tdc)
                hitID.push_back(hit)
                hitTrackID.push_back(track)
                gProcessID.push_back(proc)
                if 1 <= det <= NUM_LAYERS:
                    HitArray_mup[k, det - 1] = elem

            for elem, det, drift, tdc, hit, track, proc in zip(
                tree2.elementID,
                tree2.detectorID,
                tree2.driftDistance,
                tree2.tdcTime,
                tree2.hitID,
                tree2.hitTrackID,
                tree2.gProcessID,
            ):
                elementID.push_back(elem)
                detectorID.push_back(det)
                driftDistance.push_back(drift)
                tdcTime.push_back(tdc)
                hitID.push_back(hit)
                hitTrackID.push_back(track)
                gProcessID.push_back(proc)
                if 1 <= det <= NUM_LAYERS:
                    HitArray_mum[k, det - 1] = elem

            # per-track metadata/kinematics in order: [k μ+, k μ−]
            muID.push_back(2 * k + 1)
            muID.push_back(2 * k + 2)

            gCharge.push_back(_v(tree1.gCharge))
            gCharge.push_back(_v(tree2.gCharge))
            gTrackID.push_back(_v(tree1.gTrackID))
            gTrackID.push_back(_v(tree2.gTrackID))

            gpx.push_back(_v(tree1.gpx))
            gpy.push_back(_v(tree1.gpy))
            gpz.push_back(_v(tree1.gpz))
            gvx.push_back(_v(tree1.gvx))
            gvy.push_back(_v(tree1.gvy))
            gvz.push_back(_v(tree1.gvz))
            gpx.push_back(_v(tree2.gpx))
            gpy.push_back(_v(tree2.gpy))
            gpz.push_back(_v(tree2.gpz))
            gvx.push_back(_v(tree2.gvx))
            gvy.push_back(_v(tree2.gvy))
            gvz.push_back(_v(tree2.gvz))

        output_tree.Fill()
        fills += 1

    print(f"Fill() called {fills} times")
    print(f"Events in output tree before writing: {output_tree.GetEntries()}")

    # Write the tree only once with overwrite
    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f1.Close()
    f2.Close()

    # Verify the output
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

    # Create the output tree with CloneTree
    output_tree = tree.CloneTree(0)

    # Disable auto-save to prevent intermediate writes
    output_tree.SetAutoSave(0)  # Disable auto-save (0 or negative value)

    HitArray = np.zeros((62, 2), dtype=np.float64)
    output_tree.Branch("HitArray", HitArray, "HitArray[62][2]/D")

    fill_count = 0
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        HitArray.fill(0)

        for elem, det, drift in zip(
            tree.elementID, tree.detectorID, tree.driftDistance
        ):
            if 1 <= det <= 62:
                HitArray[det - 1][0] = elem
                HitArray[det - 1][1] = drift if elem != 0 else 0

        output_tree.Fill()
        fill_count += 1

    print(f"Fill() called {fill_count} times for {output_file}")
    print(f"Events in output tree before writing: {output_tree.GetEntries()}")

    # Write the tree only once with overwrite
    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f_in.Close()

    # Verify the output
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
        description="Combine ROOT trees and add HitArray for training."
    )
    parser.add_argument("file1", type=str, help="First ROOT file")
    parser.add_argument("file2", type=str, help="Second ROOT file")
    parser.add_argument(
        "--output", type=str, default="finder_training.root", help="Output file"
    )
    parser.add_argument(
        "--pairs", type=int, default=1, help="Number of GT dimuon pairs per event (i)"
    )

    args = parser.parse_args()

    file1_array_output = "momentum_training-1.root"
    file2_array_output = "momentum_training-2.root"
    for file in [args.output, file1_array_output, file2_array_output]:
        if os.path.exists(file):
            os.remove(file)

    print(f"ROOT version: {ROOT.gROOT.GetVersion()}")
    start_time = time.time()
    combine_files(args.file1, args.file2, args.output, n_pairs=args.pairs)
    end_time = time.time()

    start_time = time.time()
    add_hit_array(args.file1, file1_array_output)
    add_hit_array(args.file2, file2_array_output)
    end_time = time.time()

    print(f"Generated: {args.output}, {file1_array_output}, {file2_array_output}")
