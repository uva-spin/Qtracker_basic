import ROOT
import random
import argparse

# Noise settings
P_ELECTRONIC_NOISE = 0.01
P_CLUSTER_NOISE = 0.05
CLUSTER_LENGTH_RANGE = (2, 4)

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

OUTPUT_FILENAME = "noisy_output.root"  # Hardcoded output name


def inject_noise_into_event(detectorID, elementID, driftDistance, tdcTime):
    """
    Injects electronic and cluster noise, ensuring (detectorID, elementID) pairs are unique.
    Injected hits have driftDistance = 0.0 and tdcTime = 0.0 by design.
    """
    used = set((int(detectorID[i]), int(elementID[i])) for i in range(len(detectorID)))

    for det in range(NUM_DETECTORS):
        for elem in range(NUM_ELEMENT_IDS):
            if (det, elem) not in used and random.random() < P_ELECTRONIC_NOISE:
                detectorID.push_back(det)
                elementID.push_back(elem)
                driftDistance.push_back(0.0)
                tdcTime.push_back(0.0)
                used.add((det, elem))

        # Cluster noise
        if random.random() < P_CLUSTER_NOISE:
            start_elem = random.randint(0, NUM_ELEMENT_IDS - CLUSTER_LENGTH_RANGE[1] - 1)
            cluster_len = random.randint(*CLUSTER_LENGTH_RANGE)
            for offset in range(cluster_len):
                elem = start_elem + offset
                if (det, elem) not in used:
                    detectorID.push_back(det)
                    elementID.push_back(elem)
                    driftDistance.push_back(0.0)
                    tdcTime.push_back(0.0)
                    used.add((det, elem))




def inject_noise(input_file):
    fin = ROOT.TFile.Open(input_file, "READ")
    tree_in = fin.Get("tree")

    fout = ROOT.TFile.Open(OUTPUT_FILENAME, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)
    fout.cd()

    # Clone the tree *structure* (keep branch definitions)
    tree_out = tree_in.CloneTree(0)

    # Set branch addresses to modify input file's vectors
    detectorID = ROOT.std.vector("int")()
    elementID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()

    tree_in.SetBranchAddress("detectorID", detectorID)
    tree_in.SetBranchAddress("elementID", elementID)
    tree_in.SetBranchAddress("driftDistance", driftDistance)
    tree_in.SetBranchAddress("tdcTime", tdcTime)

    for i in range(tree_in.GetEntries()):
        tree_in.GetEntry(i)

        # Inject noise directly into the loaded vectors
        inject_noise_into_event(detectorID, elementID, driftDistance, tdcTime)

        # Sanity check to prevent out-of-bounds indexing later
        if not (len(detectorID) == len(elementID) == len(driftDistance) == len(tdcTime)):
            raise RuntimeError(
                f"[Noise Injection Error] Mismatch after event {i}: "
                f"det={len(detectorID)}, elem={len(elementID)}, "
                f"drift={len(driftDistance)}, tdc={len(tdcTime)}"
            )

        # Fill the modified event into output tree
        tree_out.Fill()

    fout.Write()
    fout.Close()
    fin.Close()

    print(f"[DONE] Wrote noisy output to '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject noise into a ROOT file (preserving original hits).")
    parser.add_argument("input_file", help="Path to input ROOT file")
    args = parser.parse_args()

    inject_noise(args.input_file)
