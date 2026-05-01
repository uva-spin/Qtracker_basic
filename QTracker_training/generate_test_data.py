#!/usr/bin/env python3
"""Generate minimal synthetic test data for visualization testing."""

import ROOT
import numpy as np
from array import array

def generate_synthetic_event():
    """Generate a simple synthetic single-track event."""
    # Create output file
    output_file = "data/processed_files/synthetic_test.root"
    fout = ROOT.TFile.Open(output_file, "RECREATE")
    
    # Create tree
    tree = ROOT.TTree("tree", "Synthetic test data")
    
    # Define branches
    eventID = array("i", [0])
    detectorID = ROOT.std.vector("int")()
    elementID = ROOT.std.vector("int")()
    detectorIDClean = ROOT.std.vector("int")()
    elementIDClean = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    
    # For truth tracks
    detectorID_mup = ROOT.std.vector("int")()
    elementID_mup = ROOT.std.vector("int")()
    detectorID_mum = ROOT.std.vector("int")()
    elementID_mum = ROOT.std.vector("int")()
    
    tree.Branch("eventID", eventID, "eventID/I")
    tree.Branch("detectorID", detectorID)
    tree.Branch("elementID", elementID)
    tree.Branch("detectorIDClean", detectorIDClean)
    tree.Branch("elementIDClean", elementIDClean)
    tree.Branch("driftDistance", driftDistance)
    tree.Branch("detectorID_mup", detectorID_mup)
    tree.Branch("elementID_mup", elementID_mup)
    tree.Branch("detectorID_mum", detectorID_mum)
    tree.Branch("elementID_mum", elementID_mum)
    
    # Generate a few synthetic events
    np.random.seed(42)
    
    for evt in range(10):
        eventID[0] = evt
        
        # Clear vectors
        detectorID.clear()
        elementID.clear()
        detectorIDClean.clear()
        elementIDClean.clear()
        driftDistance.clear()
        detectorID_mup.clear()
        elementID_mup.clear()
        detectorID_mum.clear()
        elementID_mum.clear()
        
        # Generate mu+ track (diagonal pattern)
        n_hits_mup = np.random.randint(20, 40)
        for i in range(n_hits_mup):
            det = min(61, max(0, int(i * 62 / n_hits_mup)))
            elem = min(200, max(0, int(100 + i * 2 + np.random.randn() * 5)))
            detectorID_mup.push_back(det)
            elementID_mup.push_back(elem)
            detectorIDClean.push_back(det)
            elementIDClean.push_back(elem)
        
        # Add noise hits
        n_noise = np.random.randint(10, 30)
        for i in range(n_noise):
            det = np.random.randint(0, 62)
            elem = np.random.randint(0, 201)
            detectorID.push_back(det)
            elementID.push_back(elem)
            driftDistance.push_back(np.random.uniform(0, 2))
        
        # Add all clean hits to noisy hits
        for i in range(detectorIDClean.size()):
            detectorID.push_back(detectorIDClean[i])
            elementID.push_back(elementIDClean[i])
            driftDistance.push_back(np.random.uniform(0, 2))
        
        tree.Fill()
    
    tree.Write()
    fout.Close()
    
    print(f"✓ Generated {output_file} with 10 synthetic events")
    print(f"  Each event has ~20-40 signal hits + 10-30 noise hits")
    return output_file

if __name__ == "__main__":
    output_file = generate_synthetic_event()
    print(f"\nTest with:")
    print(f"  python Util/visualize_single_track.py {output_file} <model.keras> --event 0 --format mp4")
