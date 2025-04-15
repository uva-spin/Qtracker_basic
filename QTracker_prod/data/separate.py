import sys
import ROOT
from array import array

# Hardcoded max number of events to process (None means use all)
MAX_EVENTS = 100000  # or set to an integer, e.g., 1000

def split_tracks(input_filename):
    input_file = ROOT.TFile.Open(input_filename, "READ")
    if not input_file or input_file.IsZombie():
        print(f"Error: Cannot open file {input_filename}")
        return

    tree = input_file.Get("tree")
    if not tree:
        print("Error: Cannot find tree named 'tree' in the file")
        input_file.Close()
        return

    n_entries = tree.GetEntries()
    if MAX_EVENTS is not None:
        n_entries = min(n_entries, MAX_EVENTS)

    output_file1 = ROOT.TFile(input_filename.replace(".root", "_track1.root"), "RECREATE")
    output_file2 = ROOT.TFile(input_filename.replace(".root", "_track2.root"), "RECREATE")

    output_file1.cd(); tree1 = ROOT.TTree("tree", "Track 1 Data")
    output_file2.cd(); tree2 = ROOT.TTree("tree", "Track 2 Data")

    # eventID is just the loop index
    eventID1 = array('i', [0])
    eventID2 = array('i', [0])
    tree1.Branch("eventID", eventID1, "eventID/I")
    tree2.Branch("eventID", eventID2, "eventID/I")

    def make_vector_branches(tree):
        vecs = {}
        def v(name, typ):
            vec = ROOT.std.vector(typ)()
            tree.Branch(name, vec)
            vecs[name] = vec
        for name in ['hitID', 'hit_trackID', 'processID', 'detectorID', 'elementID', 'gCharge', 'trackID']:
            v(name, 'int')
        for name in ['tdcTime', 'driftDistance']:
            v(name, 'double')
        for name in ['gpx', 'gpy', 'gpz', 'gvx', 'gvy', 'gvz']:
            v(name, 'double')
        return vecs

    vecs1 = make_vector_branches(tree1)
    vecs2 = make_vector_branches(tree2)

    reader = ROOT.TTreeReader(tree)

    r_hitID = ROOT.TTreeReaderArray('int')(reader, "hitID")
    r_hit_trackID = ROOT.TTreeReaderArray('int')(reader, "hit_trackID")
    r_processID = ROOT.TTreeReaderArray('int')(reader, "processID")
    r_detectorID = ROOT.TTreeReaderArray('int')(reader, "detectorID")
    r_elementID = ROOT.TTreeReaderArray('int')(reader, "elementID")
    r_tdcTime = ROOT.TTreeReaderArray('double')(reader, "tdcTime")
    r_driftDistance = ROOT.TTreeReaderArray('double')(reader, "driftDistance")
    r_gCharge = ROOT.TTreeReaderArray('int')(reader, "gCharge")
    r_trackID = ROOT.TTreeReaderArray('int')(reader, "trackID")

    r_gpx = ROOT.TTreeReaderArray('double')(reader, "gpx")
    r_gpy = ROOT.TTreeReaderArray('double')(reader, "gpy")
    r_gpz = ROOT.TTreeReaderArray('double')(reader, "gpz")
    r_gvx = ROOT.TTreeReaderArray('double')(reader, "gvx")
    r_gvy = ROOT.TTreeReaderArray('double')(reader, "gvy")
    r_gvz = ROOT.TTreeReaderArray('double')(reader, "gvz")

    for i in range(n_entries):
        reader.Next()

        # Set event ID to current index
        eventID1[0] = i
        eventID2[0] = i

        # Clear vectors
        for v in vecs1.values(): v.clear()
        for v in vecs2.values(): v.clear()

        # Track-level info
        vecs1['gCharge'].push_back(r_gCharge[0])
        vecs2['gCharge'].push_back(r_gCharge[1])
        vecs1['trackID'].push_back(r_trackID[0])
        vecs2['trackID'].push_back(r_trackID[1])

        vecs1['gpx'].push_back(r_gpx[0])
        vecs1['gpy'].push_back(r_gpy[0])
        vecs1['gpz'].push_back(r_gpz[0])
        vecs1['gvx'].push_back(r_gvx[0])
        vecs1['gvy'].push_back(r_gvy[0])
        vecs1['gvz'].push_back(r_gvz[0])

        vecs2['gpx'].push_back(r_gpx[1])
        vecs2['gpy'].push_back(r_gpy[1])
        vecs2['gpz'].push_back(r_gpz[1])
        vecs2['gvx'].push_back(r_gvx[1])
        vecs2['gvy'].push_back(r_gvy[1])
        vecs2['gvz'].push_back(r_gvz[1])

        # Assign hit-level info
        for j in range(r_hit_trackID.GetSize()):
            tid = r_hit_trackID[j]
            tgt = vecs1 if tid == 1 else vecs2 if tid == 2 else None
            if tgt:
                tgt['processID'].push_back(r_processID[j])
                tgt['hitID'].push_back(r_hitID[j])
                tgt['hit_trackID'].push_back(tid)
                tgt['detectorID'].push_back(r_detectorID[j])
                tgt['elementID'].push_back(r_elementID[j])
                tgt['driftDistance'].push_back(r_driftDistance[j])
                tgt['tdcTime'].push_back(r_tdcTime[j])

        output_file1.cd(); tree1.Fill()
        output_file2.cd(); tree2.Fill()

    output_file1.cd(); tree1.Write("", ROOT.TObject.kOverwrite); output_file1.Close()
    output_file2.cd(); tree2.Write("", ROOT.TObject.kOverwrite); output_file2.Close()
    input_file.Close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 separate.py <input.root>")
        sys.exit(1)
    split_tracks(sys.argv[1])



