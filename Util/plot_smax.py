import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201

def plot_softmax_side_by_side(root_file, event_number=0, vmax=0.01, vmin=1e-6, show_argmax=True):
    """
    Plots softmax heatmaps for mu+ and mu- side by side using log-normalized color scale.

    Args:
        root_file (str): Path to ROOT file containing softmaxTree.
        event_number (int): Event index to visualize.
        vmax (float): Maximum value for color scale.
        vmin (float): Minimum value for log scale (must be > 0).
        show_argmax (bool): Whether to overlay argmax (predicted elementID) points.
    """
    f = ROOT.TFile.Open(root_file, "READ")
    if not f or f.IsZombie():
        print(f"Error: Could not open {root_file}")
        return

    tree = f.Get("softmaxTree")
    if not tree:
        print("Error: Tree 'softmaxTree' not found in file.")
        return

    n_entries = tree.GetEntries()
    if event_number >= n_entries:
        print(f"Event {event_number} out of range (max {n_entries - 1})")
        return

    # Declare ROOT matrices
    mat_softmax_mup = ROOT.TMatrixD()
    mat_softmax_mum = ROOT.TMatrixD()

    tree.SetBranchAddress("softmax_mup", ROOT.AddressOf(mat_softmax_mup))
    tree.SetBranchAddress("softmax_mum", ROOT.AddressOf(mat_softmax_mum))
    tree.GetEntry(event_number)

    # Convert to NumPy arrays
    softmax_mup = np.array([[mat_softmax_mup[det][elem] for elem in range(NUM_ELEMENT_IDS)]
                            for det in range(NUM_DETECTORS)])
    softmax_mum = np.array([[mat_softmax_mum[det][elem] for elem in range(NUM_ELEMENT_IDS)]
                            for det in range(NUM_DETECTORS)])

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot mu+ with log scaling
    im0 = axs[0].imshow(
        softmax_mup.T,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect='auto',
        origin='lower',
        extent=[0, NUM_DETECTORS, 0, NUM_ELEMENT_IDS]
    )
    axs[0].set_title(f"Softmax Output (mu+) — Event {event_number}")
    axs[0].set_xlabel("Detector ID")
    axs[0].set_ylabel("Element ID")
    plt.colorbar(im0, ax=axs[0], label='Softmax Probability')

    # Argmax overlay (mu+)
    if show_argmax:
        argmax_mup = np.argmax(softmax_mup, axis=1)
        axs[0].scatter(np.arange(NUM_DETECTORS), argmax_mup, color='white', s=10, label="Argmax")
        axs[0].legend()

    # Plot mu- with log scaling
    im1 = axs[1].imshow(
        softmax_mum.T,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect='auto',
        origin='lower',
        extent=[0, NUM_DETECTORS, 0, NUM_ELEMENT_IDS]
    )
    axs[1].set_title(f"Softmax Output (mu−) — Event {event_number}")
    axs[1].set_xlabel("Detector ID")
    plt.colorbar(im1, ax=axs[1], label='Softmax Probability')

    # Argmax overlay (mu-)
    if show_argmax:
        argmax_mum = np.argmax(softmax_mum, axis=1)
        axs[1].scatter(np.arange(NUM_DETECTORS), argmax_mum, color='white', s=10, label="Argmax")
        axs[1].legend()

    plt.tight_layout()
    plt.show()
    f.Close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot side-by-side softmax for mu+ and mu−.")
    parser.add_argument("root_file", type=str, help="Path to ROOT file.")
    parser.add_argument("--event", type=int, default=0, help="Event number to view.")
    parser.add_argument("--vmax", type=float, default=0.01, help="Max colorbar scale.")
    parser.add_argument("--vmin", type=float, default=1e-6, help="Min colorbar scale for lognorm (must be > 0).")
    parser.add_argument("--no-argmax", action="store_true", help="Disable overlay of argmax prediction.")
    args = parser.parse_args()

    plot_softmax_side_by_side(
        args.root_file,
        event_number=args.event,
        vmax=args.vmax,
        vmin=args.vmin,
        show_argmax=not args.no_argmax
    )
