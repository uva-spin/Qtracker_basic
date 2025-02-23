import sys
import ROOT

def split_tracks(input_filename):
    # Open input ROOT file
    input_file = ROOT.TFile.Open(input_filename, "READ")
    if not input_file or input_file.IsZombie():
        print(f"Error: Cannot open file {input_filename}")
        return

    # Get the tree
    tree = input_file.Get("tree")
    if not tree:
        print("Error: Cannot find tree named 'tree' in the file")
        input_file.Close()
        return

    # Create output files
    output_file1 = ROOT.TFile(input_filename.replace(".root", "_track1.root"), "RECREATE")
    output_file2 = ROOT.TFile(input_filename.replace(".root", "_track2.root"), "RECREATE")

    # Create trees
    output_file1.cd()
    tree1 = ROOT.TTree("tree", "Track 1 Data")

    output_file2.cd()
    tree2 = ROOT.TTree("tree", "Track 2 Data")

    # Define variables for both tracks
    eventID = ROOT.std.vector('int')()
    sourceFlag = ROOT.std.vector('int')()
    processID = ROOT.std.vector('int')()

    # Track 1 variables
    gpx1, gpy1, gpz1 = ROOT.std.vector('double')(), ROOT.std.vector('double')(), ROOT.std.vector('double')()
    gvx1, gvy1, gvz1 = ROOT.std.vector('double')(), ROOT.std.vector('double')(), ROOT.std.vector('double')()
    hitID1, trackID1, detectorID1, elementID1, driftDistance1, tdcTime1 = (
        ROOT.std.vector('int')(), ROOT.std.vector('int')(),
        ROOT.std.vector('int')(), ROOT.std.vector('int')(),
        ROOT.std.vector('float')(), ROOT.std.vector('float')()
    )

    # Track 2 variables
    gpx2, gpy2, gpz2 = ROOT.std.vector('double')(), ROOT.std.vector('double')(), ROOT.std.vector('double')()
    gvx2, gvy2, gvz2 = ROOT.std.vector('double')(), ROOT.std.vector('double')(), ROOT.std.vector('double')()
    hitID2, trackID2, detectorID2, elementID2, driftDistance2, tdcTime2 = (
        ROOT.std.vector('int')(), ROOT.std.vector('int')(),
        ROOT.std.vector('int')(), ROOT.std.vector('int')(),
        ROOT.std.vector('float')(), ROOT.std.vector('float')()
    )

    # Associate branches with variables in respective trees
    tree1.Branch("eventID", eventID)
    tree1.Branch("sourceFlag", sourceFlag)
    tree1.Branch("processID", processID)
    tree1.Branch("gpx", gpx1)
    tree1.Branch("gpy", gpy1)
    tree1.Branch("gpz", gpz1)
    tree1.Branch("gvx", gvx1)
    tree1.Branch("gvy", gvy1)
    tree1.Branch("gvz", gvz1)
    tree1.Branch("hitID", hitID1)
    tree1.Branch("trackID", trackID1)
    tree1.Branch("detectorID", detectorID1)
    tree1.Branch("elementID", elementID1)
    tree1.Branch("driftDistance", driftDistance1)
    tree1.Branch("tdcTime", tdcTime1)

    tree2.Branch("eventID", eventID)
    tree2.Branch("sourceFlag", sourceFlag)
    tree2.Branch("processID", processID)
    tree2.Branch("gpx", gpx2)
    tree2.Branch("gpy", gpy2)
    tree2.Branch("gpz", gpz2)
    tree2.Branch("gvx", gvx2)
    tree2.Branch("gvy", gvy2)
    tree2.Branch("gvz", gvz2)
    tree2.Branch("hitID", hitID2)
    tree2.Branch("trackID", trackID2)
    tree2.Branch("detectorID", detectorID2)
    tree2.Branch("elementID", elementID2)
    tree2.Branch("driftDistance", driftDistance2)
    tree2.Branch("tdcTime", tdcTime2)

    # Loop over events
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        # Reset vectors
        eventID.clear()
        sourceFlag.clear()
        processID.clear()
        gpx1.clear(), gpy1.clear(), gpz1.clear()
        gvx1.clear(), gvy1.clear(), gvz1.clear()
        gpx2.clear(), gpy2.clear(), gpz2.clear()
        gvx2.clear(), gvy2.clear(), gvz2.clear()
        hitID1.clear(), trackID1.clear(), detectorID1.clear(), elementID1.clear(), driftDistance1.clear(), tdcTime1.clear()
        hitID2.clear(), trackID2.clear(), detectorID2.clear(), elementID2.clear(), driftDistance2.clear(), tdcTime2.clear()

        # Populate event metadata
        eventID.push_back(tree.eventID if not hasattr(tree.eventID, '__getitem__') else tree.eventID[0])
        sourceFlag.push_back(tree.sourceFlag[0] if hasattr(tree.sourceFlag, '__getitem__') else tree.sourceFlag)
        processID.push_back(tree.processID[0] if hasattr(tree.processID, '__getitem__') else tree.processID)

        # Read momenta and vertex positions
        gpx = list(tree.gpx)
        gpy = list(tree.gpy)
        gpz = list(tree.gpz)
        gvx = list(tree.gvx)
        gvy = list(tree.gvy)
        gvz = list(tree.gvz)

        # Process track data
        if len(gpx) >= 2:
            # Track 1
            gpx1.push_back(gpx[0]), gpy1.push_back(gpy[0]), gpz1.push_back(gpz[0])
            gvx1.push_back(gvx[0]), gvy1.push_back(gvy[0]), gvz1.push_back(gvz[0])

            # Track 2
            gpx2.push_back(gpx[1]), gpy2.push_back(gpy[1]), gpz2.push_back(gpz[1])
            gvx2.push_back(gvx[1]), gvy2.push_back(gvy[1]), gvz2.push_back(gvz[1])

        # Process hit-related variables
        for j, track in enumerate(tree.trackID):
            if track == 1:
                hitID1.push_back(tree.hitID[j])
                trackID1.push_back(track)
                detectorID1.push_back(tree.detectorID[j])
                elementID1.push_back(tree.elementID[j])
                driftDistance1.push_back(tree.driftDistance[j])
                tdcTime1.push_back(tree.tdcTime[j])
            elif track == 2:
                hitID2.push_back(tree.hitID[j])
                trackID2.push_back(track)
                detectorID2.push_back(tree.detectorID[j])
                elementID2.push_back(tree.elementID[j])
                driftDistance2.push_back(tree.driftDistance[j])
                tdcTime2.push_back(tree.tdcTime[j])

        # Fill trees
        output_file1.cd()
        tree1.Fill()

        output_file2.cd()
        tree2.Fill()

    # Write and close files
    output_file1.cd()
    tree1.Write("", ROOT.TObject.kOverwrite)
    output_file1.Close()

    output_file2.cd()
    tree2.Write("", ROOT.TObject.kOverwrite)
    output_file2.Close()

    input_file.Close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 split_tracks.py filename.root")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    split_tracks(input_filename)




