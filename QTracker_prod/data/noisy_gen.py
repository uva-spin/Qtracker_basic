import ROOT
import random
import argparse

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

OUTPUT_FILENAME = "noisy_output.root"  # Hardcoded output name


def inject_noise_into_event(detectorID, elementID, driftDistance, tdcTime, p_electronic_noise, p_cluster_noise, cluster_length_range):
    """
    Injects electronic and cluster noise, ensuring (detectorID, elementID) pairs are unique.
    Injected hits have driftDistance = 0.0 and tdcTime = 0.0 by design.
    """
    used = set((int(detectorID[i]), int(elementID[i])) for i in range(len(detectorID)))

    for det in range(NUM_DETECTORS):
        for elem in range(NUM_ELEMENT_IDS):
            if (det, elem) not in used and random.random() < p_electronic_noise:
                detectorID.push_back(det)
                elementID.push_back(elem)
                driftDistance.push_back(0.0)
                tdcTime.push_back(0.0)
                used.add((det, elem))

        # Cluster noise
        if random.random() < p_cluster_noise:
            start_elem = random.randint(0, NUM_ELEMENT_IDS - cluster_length_range[1] - 1)
            cluster_len = random.randint(*cluster_length_range)
            for offset in range(cluster_len):
                elem = start_elem + offset
                if (det, elem) not in used:
                    detectorID.push_back(det)
                    elementID.push_back(elem)
                    driftDistance.push_back(0.0)
                    tdcTime.push_back(0.0)
                    used.add((det, elem))


def inject_noise(input_file, p_electronic_noise, p_cluster_noise, cluster_length_range):
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
        inject_noise_into_event(detectorID, elementID, driftDistance, tdcTime, p_electronic_noise, p_cluster_noise, cluster_length_range)

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
    
    # Noise settings
    parser.add_argument("--p_electronic_noise", type=float, default=0.01, help="Probability of electronic noise added.")
    parser.add_argument("--p_cluster_noise", type=float, default=0.05, help="Probability of cluster noise added.")
    parser.add_argument("--cluster_length_range", type=str, default="(2,4)", help="Cluster length range.")
    args = parser.parse_args()

    CLUSTER_LENGTH_RANGE = eval(args.cluster_length_range)

    inject_noise(args.input_file, args.p_electronic_noise, args.p_cluster_noise, CLUSTER_LENGTH_RANGE)
