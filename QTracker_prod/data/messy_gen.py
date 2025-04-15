import ROOT
import numpy as np
from array import array

# Detector efficiency probability
NUM_TRACKS = 5
PROB_MEAN = 0.9
PROB_WIDTH = 0.1

# Hit fall model: "linear", "gaussian", or "exponential"
PROPAGATION_MODEL = "gaussian"
GAUSSIAN_SIGMA = 10.0
EXP_DECAY_CONST = 15.0

def inject_tracks(file1, file2, output_file, num_tracks, prob_mean, prob_width):
    if not (1 <= num_tracks <= 100):
        raise ValueError("num_tracks must be between 1 and 100.")
    if not (0 <= prob_mean <= 1):
        raise ValueError("prob_mean must be between 0 and 1.")
    if prob_width < 0:
        raise ValueError("prob_width must be non-negative.")

    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(9)
    output_tree = ROOT.TTree("tree", "Tree with injected tracks and preserved signal hit arrays")

    # Event-level and hit-level branches
    eventID = array('i', [0])
    muID = ROOT.std.vector("int")()
    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()
    hitID = ROOT.std.vector("int")()
    hit_trackID = ROOT.std.vector("int")()
    processID = ROOT.std.vector("int")()
    gCharge = ROOT.std.vector("int")()
    trackID = ROOT.std.vector("int")()
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
    output_tree.Branch("driftDistance", driftDistance)
    output_tree.Branch("tdcTime", tdcTime)
    output_tree.Branch("hitID", hitID)
    output_tree.Branch("hit_trackID", hit_trackID)
    output_tree.Branch("processID", processID)
    output_tree.Branch("trackID", trackID)
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
        driftDistance.clear()
        tdcTime.clear()
        hitID.clear()
        hit_trackID.clear()
        processID.clear()
        gCharge.clear()
        trackID.clear()
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
        trackID.push_back(tree1.trackID[0])
        trackID.push_back(tree1.trackID[1])

        for v in tree1.gpx: gpx.push_back(v)
        for v in tree1.gpy: gpy.push_back(v)
        for v in tree1.gpz: gpz.push_back(v)
        for v in tree1.gvx: gvx.push_back(v)
        for v in tree1.gvy: gvy.push_back(v)
        for v in tree1.gvz: gvz.push_back(v)

        for j in range(len(tree1.elementID)):
            elementID.push_back(tree1.elementID[j])
            detectorID.push_back(tree1.detectorID[j])
            driftDistance.push_back(tree1.driftDistance[j])
            tdcTime.push_back(tree1.tdcTime[j])
            hitID.push_back(tree1.hitID[j])
            hit_trackID.push_back(tree1.hit_trackID[j])
            processID.push_back(tree1.processID[j])

        for k in range(62):
            HitArray_mup[k] = HitArray_mup_input[k]
            HitArray_mum[k] = HitArray_mum_input[k]

        current_max_hitID = max(tree1.hitID) if len(tree1.hitID) > 0 else 0
        local_hitID_counter = current_max_hitID + 1
        next_trackID = max(tree1.trackID) + 1 if len(tree1.trackID) > 0 else 3

        for _ in range(num_tracks):
            if tree2_index >= num_events_tree2:
                break

            tree2.GetEntry(tree2_index)
            tree2_index += 1

            this_trackID = next_trackID
            next_trackID += 1
            trackID.push_back(this_trackID)
            gCharge.push_back(tree2.gCharge[0])
            gpx.push_back(tree2.gpx[0])
            gpy.push_back(tree2.gpy[0])
            gpz.push_back(tree2.gpz[0])
            gvx.push_back(tree2.gvx[0])
            gvy.push_back(tree2.gvy[0])
            gvz.push_back(tree2.gvz[0])

            probability = np.clip(np.random.normal(prob_mean, prob_width), 0, 1)

            for procID, elem, det, dist, tdc in zip(tree2.processID, tree2.elementID, tree2.detectorID, tree2.driftDistance, tree2.tdcTime):

                if PROPAGATION_MODEL == "linear":
                    weight = 1 - det / 100
                elif PROPAGATION_MODEL == "gaussian":
                    weight = np.exp(-0.5 * ((det - 1) / GAUSSIAN_SIGMA) ** 2)
                elif PROPAGATION_MODEL == "exponential":
                    weight = np.exp(-det / EXP_DECAY_CONST)
                else:
                    raise ValueError(f"Unknown PROPAGATION_MODEL: {PROPAGATION_MODEL}")

                if np.random.random() < probability * weight:
                    processID.push_back(procID)
                    elementID.push_back(elem)
                    detectorID.push_back(det)
                    driftDistance.push_back(dist)
                    tdcTime.push_back(tdc)
                    hitID.push_back(local_hitID_counter)
                    hit_trackID.push_back(this_trackID)
                    local_hitID_counter += 1

        output_tree.Fill()

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inject background tracks into ROOT data while preserving hit arrays.")
    parser.add_argument("file1", type=str, help="Path to the finder_training.root file (signal).")
    parser.add_argument("file2", type=str, help="Path to the background file.")
    parser.add_argument("--output", type=str, default="mc_events.root", help="Output ROOT file name.")
    args = parser.parse_args()

    inject_tracks(args.file1, args.file2, args.output, NUM_TRACKS, PROB_MEAN, PROB_WIDTH)
