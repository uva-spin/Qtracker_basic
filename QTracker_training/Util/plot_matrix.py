import ROOT
import numpy as np
import matplotlib.pyplot as plt

# Number of detectors and element IDs (match your constants)
NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

def plot_hit_matrices(root_filename, event_number=0):
    """
    Plots the hit matrices before and after declustering for a given event number.
    """
    f = ROOT.TFile.Open(root_filename, "READ")
    if not f or f.IsZombie():
        print(f"Error: Could not open {root_filename}")
        return

    tree = f.Get("hitMatrixTree")
    if not tree:
        print("Error: Tree 'hitMatrixTree' not found in the file.")
        return

    n_entries = tree.GetEntries()
    if event_number >= n_entries:
        print(f"Error: Event number {event_number} out of range (only {n_entries} events).")
        return

    # Set branch addresses
    mat_before = ROOT.TMatrixD()
    mat_after = ROOT.TMatrixD()
    tree.SetBranchAddress("hitMatrix_before", mat_before)
    tree.SetBranchAddress("hitMatrix_after", mat_after)

    # Get the requested event
    tree.GetEntry(event_number)

    # Convert TMatrixD to numpy arrays
    before = np.zeros((NUM_DETECTORS, NUM_ELEMENT_IDS))
    after = np.zeros((NUM_DETECTORS, NUM_ELEMENT_IDS))
    
    for i in range(NUM_DETECTORS):
        for j in range(NUM_ELEMENT_IDS):
            before[i, j] = mat_before[i][j]
            after[i, j] = mat_after[i][j]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axs[0].imshow(before.T, aspect='auto', origin='lower')  # <-- .T here
    axs[0].set_title(f"Before Declusterizing (Event {event_number})")
    axs[0].set_xlabel("Detector ID (x-axis)")
    axs[0].set_ylabel("Element ID (y-axis)")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(after.T, aspect='auto', origin='lower')  # <-- .T here
    axs[1].set_title(f"After Declusterizing (Event {event_number})")
    axs[1].set_xlabel("Detector ID (x-axis)")
    axs[1].set_ylabel("Element ID (y-axis)")
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.show()

    f.Close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot before and after hit matrices from ROOT file.")
    parser.add_argument("root_file", type=str, help="Path to the ROOT output file (e.g., qtracker_reco.root)")
    parser.add_argument("--event", type=int, default=0, help="Event number to plot (default=0)")
    args = parser.parse_args()

    plot_hit_matrices(args.root_file, args.event)
