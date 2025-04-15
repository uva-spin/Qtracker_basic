import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse

def calculate_invariant_mass(tlv1, tlv2):
    """Computes the invariant mass of two particles using TLorentzVectors."""
    return (tlv1 + tlv2).M()

def extract_momenta_and_calculate_mass(root_file):
    """Extracts momenta from ROOT file and computes invariant mass using TLorentzVectors."""
    f = ROOT.TFile.Open(root_file, "READ")
    tree = f.Get("tree")

    if not tree:
        print("Error: Tree not found in file.")
        return np.array([])

    masses = []
    
    for event in tree:
        tlv_mup = ROOT.TLorentzVector()
        tlv_mum = ROOT.TLorentzVector()

        for i in range(len(event.muID)):
            if event.muID[i] == 1:  # mu+
                tlv_mup.SetPxPyPzE(event.qpx[i], event.qpy[i], event.qpz[i], 
                                   np.sqrt(event.qpx[i]**2 + event.qpy[i]**2 + event.qpz[i]**2 + 0.105**2))
            elif event.muID[i] == 2:  # mu-
                tlv_mum.SetPxPyPzE(event.qpx[i], event.qpy[i], event.qpz[i], 
                                   np.sqrt(event.qpx[i]**2 + event.qpy[i]**2 + event.qpz[i]**2 + 0.105**2))
        
        # Check if TLorentzVectors are initialized (non-zero energy)
        if tlv_mup.E() > 0 and tlv_mum.E() > 0:
            mass = calculate_invariant_mass(tlv_mup, tlv_mum)
            masses.append(mass)
    
    f.Close()
    return np.array(masses)

def plot_invariant_mass(masses, output_file="invariant_mass.png"):
    plt.figure(figsize=(8,6))
    plt.hist(masses, bins=np.linspace(0, 6, 300), alpha=0.7, color='b', edgecolor='black')
    plt.xlabel("Invariant Mass (GeV/c^2)")
    plt.ylabel("Count")
    plt.title("Invariant Mass Distribution of Muon Pairs")
    plt.xlim(0, 6)  # Keep full range
    plt.grid()
    plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot the invariant mass of muon pairs from a ROOT file.")
    parser.add_argument("root_file", type=str, help="Path to the ROOT file containing inferred momenta.")
    parser.add_argument("--output_plot", type=str, default="invariant_mass.png", help="Output file for the invariant mass plot.")
    args = parser.parse_args()
    
    masses = extract_momenta_and_calculate_mass(args.root_file)
    
    if masses.size > 0:
        plot_invariant_mass(masses, args.output_plot)
    else:
        print("No valid muon pairs found for invariant mass calculation.")

if __name__ == "__main__":
    main()
