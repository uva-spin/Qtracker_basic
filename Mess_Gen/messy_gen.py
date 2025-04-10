import ROOT
import numpy as np
from array import array


# Detector efficiency probability
NUM_TRACKS = 50            # Number of tracks to inject (1-100)
PROB_MEAN = 0.9           # Mean probability for keeping hits
PROB_WIDTH = 0.1          # Width of probability distribution

# Hit fall model: "linear", "gaussian", or "exponential"
PROPAGATION_MODEL = "gaussian"

# Parameters for propagation models
GAUSSIAN_SIGMA = 10.0     # For Gaussian model, sigma of tail
EXP_DECAY_CONST = 15.0    # For Exponential model, decay constant


def inject_tracks(file1, file2, output_file, num_tracks, prob_mean, prob_width):
    if num_tracks < 1 or num_tracks > 100:
        raise ValueError("The number of tracks to inject must be between 1 and 100.")
    if not (0 <= prob_mean <= 1):
        raise ValueError("The mean probability must be between 0 and 1.")
    if prob_width < 0:
        raise ValueError("The width of the probability distribution must be non-negative.")

    # Open input files
    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")

    # Access trees
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    # Output file
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(9)

    output_tree = ROOT.TTree("tree", "Tree with injected tracks and additional data")
    output_tree.SetAutoFlush(0)

    # Prepare branches
    eventID = array('i', [0])
    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()
    hitID = ROOT.std.vector("int")()
    processID = ROOT.std.vector("int")()
    trackID = ROOT.std.vector("int")()
    gCharge = ROOT.std.vector("int")()
    gvx = ROOT.std.vector("double")()
    gvy = ROOT.std.vector("double")()
    gvz = ROOT.std.vector("double")()
    gpx = ROOT.std.vector("double")()
    gpy = ROOT.std.vector("double")()
    gpz = ROOT.std.vector("double")()

    output_tree.Branch("eventID", eventID, "eventID/I")
    output_tree.Branch("elementID", elementID)
    output_tree.Branch("detectorID", detectorID)
    output_tree.Branch("driftDistance", driftDistance)
    output_tree.Branch("tdcTime", tdcTime)
    output_tree.Branch("hitID", hitID)
    output_tree.Branch("processID", processID)
    output_tree.Branch("trackID", trackID)
    output_tree.Branch("gCharge", gCharge)
    output_tree.Branch("gvx", gvx)
    output_tree.Branch("gvy", gvy)
    output_tree.Branch("gvz", gvz)
    output_tree.Branch("gpx", gpx)
    output_tree.Branch("gpy", gpy)
    output_tree.Branch("gpz", gpz)

    num_events_tree2 = tree2.GetEntries()
    tree2_index = 0

    for i, entry1 in enumerate(tree1):
        if tree2_index >= num_events_tree2:
            break

        # Reset vectors
        elementID.clear()
        detectorID.clear()
        driftDistance.clear()
        tdcTime.clear()
        hitID.clear()
        processID.clear()
        trackID.clear()
        gCharge.clear()
        gvx.clear()
        gvy.clear()
        gvz.clear()
        gpx.clear()
        gpy.clear()
        gpz.clear()

        eventID[0] = i

        # Hits from original event
        for j in range(len(entry1.elementID)):
            elementID.push_back(entry1.elementID[j])
            detectorID.push_back(entry1.detectorID[j])
            driftDistance.push_back(entry1.driftDistance[j])
            tdcTime.push_back(entry1.tdcTime[j])
            hitID.push_back(entry1.hitID[j])
            processID.push_back(entry1.processID[j])

        # Track-level info (assumed 2 tracks)
        trackID.push_back(1)
        trackID.push_back(2)
        gCharge.push_back(entry1.gCharge[0])
        gCharge.push_back(entry1.gCharge[1])
        gvx.push_back(entry1.gvx[0])
        gvx.push_back(entry1.gvx[1])
        gvy.push_back(entry1.gvy[0])
        gvy.push_back(entry1.gvy[1])
        gvz.push_back(entry1.gvz[0])
        gvz.push_back(entry1.gvz[1])
        gpx.push_back(entry1.gpx[0])
        gpx.push_back(entry1.gpx[1])
        gpy.push_back(entry1.gpy[0])
        gpy.push_back(entry1.gpy[1])
        gpz.push_back(entry1.gpz[0])
        gpz.push_back(entry1.gpz[1])

        current_max_hitID = max(entry1.hitID) if len(entry1.hitID) > 0 else 0
        local_hitID_counter = current_max_hitID + 1
        next_trackID = 3

        for _ in range(num_tracks):
            if tree2_index >= num_events_tree2:
                break

            tree2.GetEntry(tree2_index)
            tree2_index += 1

            this_trackID = next_trackID
            next_trackID += 1

            trackID.push_back(this_trackID)
            gCharge.push_back(tree2.gCharge[0])
            gvx.push_back(tree2.gvx[0])
            gvy.push_back(tree2.gvy[0])
            gvz.push_back(tree2.gvz[0])
            gpx.push_back(tree2.gpx[0])
            gpy.push_back(tree2.gpy[0])
            gpz.push_back(tree2.gpz[0])

            probability = np.clip(np.random.normal(prob_mean, prob_width), 0, 1)

            for procID, elem, det, dist, tdc in zip(tree2.processID, tree2.elementID, tree2.detectorID,
                                                    tree2.driftDistance, tree2.tdcTime):

                # Propagation model logic
                if PROPAGATION_MODEL == "linear":
                    weight = 1 - det / 100
                elif PROPAGATION_MODEL == "gaussian":
                    weight = np.exp(-0.5 * ((det - 1) / GAUSSIAN_SIGMA) ** 2)
                elif PROPAGATION_MODEL == "exponential":
                    weight = np.exp(-det / EXP_DECAY_CONST)
                else:
                    raise ValueError(f"Unknown PROPAGATION_MODEL: {PROPAGATION_MODEL}")

                keep_hit = np.random.random() < probability * weight

                if keep_hit:
                    processID.push_back(procID)
                    elementID.push_back(elem)
                    detectorID.push_back(det)
                    driftDistance.push_back(dist)
                    tdcTime.push_back(tdc)
                    hitID.push_back(local_hitID_counter)
                    local_hitID_counter += 1

        output_tree.Fill()

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inject tracks with a hit-keeping probability into a ROOT tree.")
    parser.add_argument("file1", type=str, help="Path to the first ROOT file.")
    parser.add_argument("file2", type=str, help="Path to the second ROOT file.")
    parser.add_argument("--output", type=str, default="mc_events.root", help="Output ROOT file name.")
    args = parser.parse_args()

    inject_tracks(args.file1, args.file2, args.output, NUM_TRACKS, PROB_MEAN, PROB_WIDTH)
