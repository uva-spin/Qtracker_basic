import ROOT
import argparse
import numpy as np
from array import array
import time
import os



def combine_files(file1, file2, output_file):

    if os.path.exists(output_file):
        os.remove(output_file)

    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")

    print(f"Entries in {file1}: {tree1.GetEntries()}")
    print(f"Entries in {file2}: {tree2.GetEntries()}")
    print("Trees in file1:", [key.GetName() for key in f1.GetListOfKeys() if key.GetClassName() == "TTree"])
    print("Trees in file2:", [key.GetName() for key in f2.GetListOfKeys() if key.GetClassName() == "TTree"])
    
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")
    
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)

    # Create the output tree
    output_tree = ROOT.TTree("tree", "Tree with combined hits and track information")
    
    output_tree.SetAutoSave(0)  # Disable auto-save (0 or negative value)
    
    eventID = array('i', [0])
    muID = ROOT.std.vector("int")()

    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()
    hitID = ROOT.std.vector("int")()
    hitTrackID = ROOT.std.vector("int")()
    gProcessID = ROOT.std.vector("int")()
    gCharge = ROOT.std.vector('int')()   
    gTrackID = ROOT.std.vector('int')()   

    gpx = ROOT.std.vector("double")()
    gpy = ROOT.std.vector("double")()
    gpz = ROOT.std.vector("double")()
    gvx = ROOT.std.vector("double")()
    gvy = ROOT.std.vector("double")()
    gvz = ROOT.std.vector("double")()

    HitArray_mup = np.zeros(62, dtype=np.int32)
    HitArray_mum = np.zeros(62, dtype=np.int32)

    output_tree.Branch("eventID", eventID, "eventID/I")
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
    output_tree.Branch("HitArray_mup", HitArray_mup, "HitArray_mup[62]/I")
    output_tree.Branch("HitArray_mum", HitArray_mum, "HitArray_mum[62]/I")
    
    fill_count = 0
    for i in range(min(tree1.GetEntries(), tree2.GetEntries())):
        tree1.GetEntry(i)
        tree2.GetEntry(i)

        # Clear all vectors before filling
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

        # index event
        eventID[0] = i

        # Assign mu+ and mu- IDs correctly
        muID.push_back(1)  # mu+ track
        muID.push_back(2)  # mu- track

        gCharge.push_back(tree1.gCharge[0] if hasattr(tree1.gCharge, '__getitem__') else tree1.gCharge)
        gCharge.push_back(tree2.gCharge[0] if hasattr(tree2.gCharge, '__getitem__') else tree2.gCharge)
        gTrackID.push_back(tree1.gTrackID[0] if hasattr(tree1.gTrackID, '__getitem__') else tree1.gTrackID)
        gTrackID.push_back(tree2.gTrackID[0] if hasattr(tree2.gTrackID, '__getitem__') else tree2.gTrackID)
        
        # Process mu+ hits
        for elem, det, drift, tdc, hit, track, proc in zip(
            tree1.elementID, tree1.detectorID, tree1.driftDistance, tree1.tdcTime,
            tree1.hitID, tree1.hitTrackID, tree1.gProcessID
        ):      
            elementID.push_back(elem)
            detectorID.push_back(det)
            driftDistance.push_back(drift)
            tdcTime.push_back(tdc)
            hitID.push_back(hit)
            hitTrackID.push_back(track)
            gProcessID.push_back(proc)

            if 1 <= det <= 62:  # Ensure valid indexing
                HitArray_mup[det - 1] = elem

        # Process mu- hits
        for elem, det, drift, tdc, hit, track, proc in zip(
            tree2.elementID, tree2.detectorID, tree2.driftDistance, tree2.tdcTime,
            tree2.hitID, tree2.hitTrackID, tree2.gProcessID
        ):
            elementID.push_back(elem)
            detectorID.push_back(det)
            driftDistance.push_back(drift)
            tdcTime.push_back(tdc)
            hitID.push_back(hit)
            hitTrackID.push_back(track)
            gProcessID.push_back(proc)

            if 1 <= det <= 62:  # Ensure valid indexing
                HitArray_mum[det - 1] = elem

        # Assign gpx, gpy, gpz, gvx, gvy, gvz event-by-event
        # First track: mu+
        gpx.push_back(tree1.gpx[0])
        gpy.push_back(tree1.gpy[0])
        gpz.push_back(tree1.gpz[0])
        gvx.push_back(tree1.gvx[0])
        gvy.push_back(tree1.gvy[0])
        gvz.push_back(tree1.gvz[0])

        # Second track: mu-
        gpx.push_back(tree2.gpx[0])
        gpy.push_back(tree2.gpy[0])
        gpz.push_back(tree2.gpz[0])
        gvx.push_back(tree2.gvx[0])
        gvy.push_back(tree2.gvy[0])
        gvz.push_back(tree2.gvz[0])

        output_tree.Fill()
        fill_count += 1

    print(f"Fill() called {fill_count} times")
    print(f"Events in output tree before writing: {output_tree.GetEntries()}")
    
    # Write the tree only once with overwrite
    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f1.Close()
    f2.Close()

    fout = ROOT.TFile.Open(output_file, "READ")
    out_tree = fout.Get("tree")
    print(f"Events in output tree after writing: {out_tree.GetEntries()}")
    print("Trees in output file:", [key.GetName() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"])
    cycles = [key.GetCycle() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"]
    print("Tree cycles in output file:", cycles)
    fout.Close()


def add_hit_array(input_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    f_in = ROOT.TFile.Open(input_file, "READ")
    tree = f_in.Get("tree")

    print(f"Entries in input file {input_file}: {tree.GetEntries()}")
    print("Trees in input file:", [key.GetName() for key in f_in.GetListOfKeys() if key.GetClassName() == "TTree"])
    
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)

    # Create the output tree with CloneTree
    output_tree = tree.CloneTree(0)
    
    # Change HitArray to 3D: 62 elements for elementID and 62 elements for driftDistance
    # Disable auto-save to prevent intermediate writes
    output_tree.SetAutoSave(0)  # Disable auto-save (0 or negative value)
    
    HitArray = np.zeros((62, 2), dtype=np.float64)
    output_tree.Branch("HitArray", HitArray, "HitArray[62][2]/D")

    fill_count = 0
    
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        HitArray.fill(0)  # Reset HitArray
        
        for elem, det, drift in zip(tree.elementID, tree.detectorID, tree.driftDistance):
            if 1 <= det <= 62:
                HitArray[det - 1][0] = elem  # Store elementID
                HitArray[det - 1][1] = drift if elem != 0 else 0  # Store driftDistance, set to 0 if elementID is 0
        
        output_tree.Fill()
        fill_count += 1
    
    print(f"Fill() called {fill_count} times for {output_file}")
    print(f"Events in output tree before writing: {output_tree.GetEntries()}")
    
    output_tree.Write("", ROOT.TObject.kOverwrite)
    fout.Close()
    f_in.Close()

    fout = ROOT.TFile.Open(output_file, "READ")
    out_tree = fout.Get("tree")
    print(f"Events in output tree after writing: {out_tree.GetEntries()}")
    print("Trees in output file:", [key.GetName() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"])
    cycles = [key.GetCycle() for key in fout.GetListOfKeys() if key.GetClassName() == "TTree"]
    print("Tree cycles in output file:", cycles)
    fout.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine ROOT trees and add HitArray for training.")
    parser.add_argument("file1", type=str, help="First ROOT file")
    parser.add_argument("file2", type=str, help="Second ROOT file")
    parser.add_argument("--output", type=str, default="finder_training.root", help="Output file")

    args = parser.parse_args()

    file1_array_output = "momentum_training-1.root"
    file2_array_output = "momentum_training-2.root"
    for file in [args.output, file1_array_output, file2_array_output]:
        if os.path.exists(file):
            os.remove(file)

    print(f"ROOT version: {ROOT.gROOT.GetVersion()}")

    start_time = time.time()
    combine_files(args.file1, args.file2, args.output)
    end_time = time.time()
    #print(f"[TIMER] combine_files() completed in {end_time - start_time:.2f} seconds")


    start_time = time.time()
    add_hit_array(args.file1, file1_array_output)
    add_hit_array(args.file2, file2_array_output)
    end_time = time.time()
    #print(f"[TIMER] add_hit_array() (both files) completed in {end_time - start_time:.2f} seconds")

    print(f"Generated: {args.output}, {file1_array_output}, {file2_array_output}")
