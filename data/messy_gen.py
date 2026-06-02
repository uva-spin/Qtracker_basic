import ROOT
import numpy as np
from array import array
import random


# Detector efficiency probability
NUM_TRACKS = 50
PROB_MEAN = 0.9
PROB_WIDTH = 0.1

# Hit fall model: "linear", "gaussian", or "exponential"
PROPAGATION_MODEL = "gaussian"
GAUSSIAN_SIGMA = 60.0
EXP_DECAY_CONST = 15.0


def inject_tracks(
    file1, file2, output_file, num_tracks, lower_bound=0, uniform_tracks=True
):
    if not (1 <= num_tracks <= 100):
        raise ValueError("num_tracks must be between 1 and 100.")
    if not (0 <= PROB_MEAN <= 1):
        raise ValueError("PROB_MEAN must be between 0 and 1.")
    if PROB_WIDTH < 0:
        raise ValueError("PROB_WIDTH must be non-negative.")

    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(9)
    output_tree = ROOT.TTree(
        "tree", "Tree with injected tracks and preserved signal hit arrays"
    )

    # Event-level and hit-level branches
    eventID = array("i", [0])
    muID = ROOT.std.vector("int")()
    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    elementIDClean = ROOT.std.vector("int")()
    detectorIDClean = ROOT.std.vector("int")()
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

    HitArray_mup = np.zeros(62, dtype=np.int32)
    HitArray_mum = np.zeros(62, dtype=np.int32)

    # Output tree branches
    output_tree.Branch("eventID", eventID, "eventID/I")
    output_tree.Branch("muID", muID)
    output_tree.Branch("elementID", elementID)
    output_tree.Branch("detectorID", detectorID)
    output_tree.Branch("elementIDClean", elementIDClean)
    output_tree.Branch("detectorIDClean", detectorIDClean)
    output_tree.Branch("driftDistance", driftDistance)
    output_tree.Branch("tdcTime", tdcTime)
    output_tree.Branch("hitID", hitID)
    output_tree.Branch("hitTrackID", hitTrackID)
    output_tree.Branch("gProcessID", gProcessID)
    output_tree.Branch("gTrackID", gTrackID)
    output_tree.Branch("gCharge", gCharge)
    output_tree.Branch("gpx", gpx)
    output_tree.Branch("gpy", gpy)
    output_tree.Branch("gpz", gpz)
    output_tree.Branch("gvx", gvx)
    output_tree.Branch("gvy", gvy)
    output_tree.Branch("gvz", gvz)
    output_tree.Branch("HitArray_mup", HitArray_mup, "HitArray_mup[62]/I")
    output_tree.Branch("HitArray_mum", HitArray_mum, "HitArray_mum[62]/I")

    # Bind input HitArray branches to NumPy buffers
    HitArray_mup_input = np.zeros(62, dtype=np.int32)
    HitArray_mum_input = np.zeros(62, dtype=np.int32)
    tree1.SetBranchAddress("HitArray_mup", HitArray_mup_input)
    tree1.SetBranchAddress("HitArray_mum", HitArray_mum_input)

    num_events_tree2 = tree2.GetEntries()
    tree2_index = 0

    for i in range(tree1.GetEntries()):
        if tree2_index >= num_events_tree2:
            break

        tree1.GetEntry(i)
        tree2.GetEntry(tree2_index)
        tree2_index += 1

        # Reset
        eventID[0] = tree1.eventID
        muID.clear()
        elementID.clear()
        detectorID.clear()
        elementIDClean.clear()
        detectorIDClean.clear()
        driftDistance.clear()
        tdcTime.clear()
        hitID.clear()
        hitTrackID.clear()
        gProcessID.clear()
        gCharge.clear()
        gTrackID.clear()
        gpx.clear()
        gpy.clear()
        gpz.clear()
        gvx.clear()
        gvy.clear()
        gvz.clear()
        HitArray_mup.fill(0)
        HitArray_mum.fill(0)

        # Preserve muID, gCharge, trackID (true signal)
        muID.push_back(1)
        muID.push_back(2)
        gCharge.push_back(tree1.gCharge[0])
        gCharge.push_back(tree1.gCharge[1])
        gTrackID.push_back(tree1.gTrackID[0])
        gTrackID.push_back(tree1.gTrackID[1])

        for v in tree1.gpx:
            gpx.push_back(v)
        for v in tree1.gpy:
            gpy.push_back(v)
        for v in tree1.gpz:
            gpz.push_back(v)
        for v in tree1.gvx:
            gvx.push_back(v)
        for v in tree1.gvy:
            gvy.push_back(v)
        for v in tree1.gvz:
            gvz.push_back(v)

        for j in range(len(tree1.elementID)):
            elementID.push_back(tree1.elementID[j])
            detectorID.push_back(tree1.detectorID[j])
            elementIDClean.push_back(tree1.elementID[j])
            detectorIDClean.push_back(tree1.detectorID[j])
            driftDistance.push_back(tree1.driftDistance[j])
            tdcTime.push_back(tree1.tdcTime[j])
            hitID.push_back(tree1.hitID[j])
            hitTrackID.push_back(tree1.hitTrackID[j])
            gProcessID.push_back(tree1.gProcessID[j])

        for k in range(62):
            HitArray_mup[k] = HitArray_mup_input[k]
            HitArray_mum[k] = HitArray_mum_input[k]

        current_max_hitID = max(tree1.hitID) if len(tree1.hitID) > 0 else 0
        local_hitID_counter = current_max_hitID + 1
        next_trackID = max(tree1.gTrackID) + 1 if len(tree1.gTrackID) > 0 else 3

        weights = [2] * (num_tracks - lower_bound + 1)

        # Skew probabilities towards higher track counts
        if not uniform_tracks:
            for i in range(len(weights)):
                if i >= len(weights) // 3:
                    weights[i] += 1
                if i >= len(weights) // 3 * 2:
                    weights[i] += 1

        random_num_tracks = random.choices(
            range(lower_bound, num_tracks + 1), weights=weights, k=1
        )[0]
        # random_num_tracks = random.randint(lower_bound, num_tracks)

        for _ in range(random_num_tracks):
            if tree2_index >= num_events_tree2:
                break

            tree2.GetEntry(tree2_index)
            tree2_index += 1

            this_trackID = next_trackID
            next_trackID += 1
            gTrackID.push_back(this_trackID)
            gCharge.push_back(tree2.gCharge[0])
            gpx.push_back(tree2.gpx[0])
            gpy.push_back(tree2.gpy[0])
            gpz.push_back(tree2.gpz[0])
            gvx.push_back(tree2.gvx[0])
            gvy.push_back(tree2.gvy[0])
            gvz.push_back(tree2.gvz[0])

            probability = np.clip(np.random.normal(PROB_MEAN, PROB_WIDTH), 0, 1)

            for procID, elem, det, dist, tdc in zip(
                tree2.gProcessID,
                tree2.elementID,
                tree2.detectorID,
                tree2.driftDistance,
                tree2.tdcTime,
            ):
                if PROPAGATION_MODEL == "linear":
                    weight = 1 - det / 100
                elif PROPAGATION_MODEL == "gaussian":
                    weight = np.exp(-0.5 * ((det - 1) / GAUSSIAN_SIGMA) ** 2)
                elif PROPAGATION_MODEL == "exponential":
                    weight = np.exp(-det / EXP_DECAY_CONST)
                else:
                    raise ValueError(f"Unknown PROPAGATION_MODEL: {PROPAGATION_MODEL}")

                if np.random.random() < probability * weight:
                    gProcessID.push_back(procID)
                    elementID.push_back(elem)
                    detectorID.push_back(det)
                    driftDistance.push_back(dist)
                    tdcTime.push_back(tdc)
                    hitID.push_back(local_hitID_counter)
                    hitTrackID.push_back(this_trackID)
                    local_hitID_counter += 1

        output_tree.Fill()

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inject background tracks into ROOT data while preserving hit arrays."
    )
    parser.add_argument(
        "file1", type=str, help="Path to the finder_training.root file (signal)."
    )
    parser.add_argument("file2", type=str, help="Path to the background file.")
    parser.add_argument(
        "--output", type=str, default="mc_events.root", help="Output ROOT file name."
    )
    parser.add_argument(
        "--num_tracks",
        type=int,
        default=NUM_TRACKS,
        help="Upper bound number of background tracks to inject.",
    )
    parser.add_argument(
        "--lower_bound",
        type=int,
        default=0,
        help="Lower bound number of background tracks to inject.",
    )
    parser.add_argument(
        "--uniform_tracks",
        type=int,
        default=1,
        help="Use uniform distribution for track injection: 1 for True, 0 for False.",
    )
    args = parser.parse_args()

    inject_tracks(
        args.file1,
        args.file2,
        args.output,
        args.num_tracks,
        lower_bound=args.lower_bound,
        uniform_tracks=bool(args.uniform_tracks),
    )
