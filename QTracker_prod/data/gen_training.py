import ROOT
import argparse
import numpy as np
from array import array
import time



def combine_files(file1, file2, output_file):
    f1 = ROOT.TFile.Open(file1, "READ")
    f2 = ROOT.TFile.Open(file2, "READ")
    
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")
    
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)
    output_tree = ROOT.TTree("tree", "Tree with combined hits and track information")
    
    eventID = ROOT.Int_t(0)

    muID = ROOT.std.vector("int")()  # Store mu+ (1) and mu- (2) for each event

    elementID = ROOT.std.vector("int")()
    detectorID = ROOT.std.vector("int")()
    driftDistance = ROOT.std.vector("double")()
    tdcTime = ROOT.std.vector("double")()
    hitID = ROOT.std.vector("int")()
    hit_trackID = ROOT.std.vector("int")()
    processID = ROOT.std.vector("int")()
    gCharge = ROOT.std.vector('int')()   
    trackID = ROOT.std.vector('int')()   

    gpx = ROOT.std.vector("double")()
    gpy = ROOT.std.vector("double")()
    gpz = ROOT.std.vector("double")()
    gvx = ROOT.std.vector("double")()
    gvy = ROOT.std.vector("double")()
    gvz = ROOT.std.vector("double")()

    HitArray_mup = np.zeros(62, dtype=np.int32)
    HitArray_mum = np.zeros(62, dtype=np.int32)
    eventID = array('i', [0])
    output_tree.Branch("eventID", eventID, "eventID/I")
    output_tree.Branch("muID", muID)  
    output_tree.Branch("elementID", elementID)
    output_tree.Branch("detectorID", detectorID)
    output_tree.Branch("driftDistance", driftDistance)
    output_tree.Branch("tdcTime", tdcTime)
    output_tree.Branch("hitID", hitID)
    output_tree.Branch("hit_trackID", hit_trackID)
    output_tree.Branch("processID", processID)
    output_tree.Branch("gCharge", gCharge)
    output_tree.Branch("trackID", trackID)
    output_tree.Branch("gpx", gpx)
    output_tree.Branch("gpy", gpy)
    output_tree.Branch("gpz", gpz)
    output_tree.Branch("gvx", gvx)
    output_tree.Branch("gvy", gvy)
    output_tree.Branch("gvz", gvz)
    output_tree.Branch("HitArray_mup", HitArray_mup, "HitArray_mup[62]/I")
    output_tree.Branch("HitArray_mum", HitArray_mum, "HitArray_mum[62]/I")
    

    for i in range(min(tree1.GetEntries(), tree2.GetEntries())):
        tree1.GetEntry(i)
        tree2.GetEntry(i)

        # Clear all vectors before filling
        muID.clear()
        elementID.clear()
        detectorID.clear()
        driftDistance.clear()
        processID.clear()       
        tdcTime.clear()
        hitID.clear()
        hit_trackID.clear()
        gpx.clear()
        gpy.clear()
        gpz.clear()
        gvx.clear()
        gvy.clear()
        gvz.clear()
        gCharge.clear()
        trackID.clear()
        HitArray_mup.fill(0)
        HitArray_mum.fill(0)

        # index event
        eventID[0] = i

        # Assign mu+ and mu- IDs correctly
        muID.push_back(1)  # mu+ track
        muID.push_back(2)  # mu- track

        gCharge.push_back(tree1.gCharge[0] if hasattr(tree1.gCharge, '__getitem__') else tree1.gCharge)
        gCharge.push_back(tree2.gCharge[0] if hasattr(tree2.gCharge, '__getitem__') else tree2.gCharge)
        trackID.push_back(tree1.trackID[0] if hasattr(tree1.trackID, '__getitem__') else tree1.trackID)
        trackID.push_back(tree2.trackID[0] if hasattr(tree2.trackID, '__getitem__') else tree2.trackID)
        
        # Process mu+ hits
        for elem, det, drift, tdc, hit, track, proc in zip(tree1.elementID, tree1.detectorID, tree1.driftDistance, tree1.tdcTime, tree1.hitID, tree1.hit_trackID, tree1.processID):      
            elementID.push_back(elem)
            detectorID.push_back(det)
            driftDistance.push_back(drift)
            tdcTime.push_back(tdc)
            hitID.push_back(hit)
            hit_trackID.push_back(track)
            processID.push_back(proc)

            if 1 <= det <= 62:  # Ensure valid indexing
                HitArray_mup[det - 1] = elem

        # Process mu- hits
        for elem, det, drift, tdc, hit, track, proc in zip(tree2.elementID, tree2.detectorID, tree2.driftDistance, tree2.tdcTime, tree2.hitID, tree2.hit_trackID, tree2.processID):    
            elementID.push_back(elem)
            detectorID.push_back(det)
            driftDistance.push_back(drift)
            tdcTime.push_back(tdc)
            hitID.push_back(hit)
            hit_trackID.push_back(track)
            processID.push_back(proc)

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

    fout.Write()
    fout.Close()
    f1.Close()
    f2.Close()


def add_hit_array(input_file, output_file):
    f_in = ROOT.TFile.Open(input_file, "READ")
    tree = f_in.Get("tree")
    
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(5)
    output_tree = tree.CloneTree(0)
    
    # Change HitArray to 3D: 62 elements for elementID and 62 elements for driftDistance
    HitArray = np.zeros((62, 2), dtype=np.float64)  # First column for elementID, second for driftDistance
    output_tree.Branch("HitArray", HitArray, "HitArray[62][2]/D")
    
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        HitArray.fill(0)  # Reset HitArray
        
        for elem, det, drift in zip(tree.elementID, tree.detectorID, tree.driftDistance):
            if 1 <= det <= 62:
                HitArray[det - 1][0] = elem  # Store elementID
                HitArray[det - 1][1] = drift if elem != 0 else 0  # Store driftDistance, set to 0 if elementID is 0
        
        output_tree.Fill()
    
    fout.Write()
    fout.Close()
    f_in.Close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine ROOT trees and add HitArray for training.")
    parser.add_argument("file1", type=str, help="First ROOT file")
    parser.add_argument("file2", type=str, help="Second ROOT file")
    parser.add_argument("--output", type=str, default="finder_training.root", help="Output file")

    args = parser.parse_args()

    start_time = time.time()
    combine_files(args.file1, args.file2, args.output)
    end_time = time.time()
    #print(f"[TIMER] combine_files() completed in {end_time - start_time:.2f} seconds")

    file1_array_output = "momentum_training-1.root"
    file2_array_output = "momentum_training-2.root"

    start_time = time.time()
    add_hit_array(args.file1, file1_array_output)
    add_hit_array(args.file2, file2_array_output)
    end_time = time.time()
    #print(f"[TIMER] add_hit_array() (both files) completed in {end_time - start_time:.2f} seconds")

    print(f"Generated: {args.output}, {file1_array_output}, {file2_array_output}")
