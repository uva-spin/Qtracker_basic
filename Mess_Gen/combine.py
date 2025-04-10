import ROOT
import argparse
import signal
import sys
import array

# Set this to limit total events written. Set to None to go until one file ends.
MAX_OUTPUT_EVENTS = 10000  # e.g., 200000 for a hard cutoff, or None for full merge

def merge_alternating_reader(file1, file2, output_file):
    f1 = ROOT.TFile.Open(file1)
    f2 = ROOT.TFile.Open(file2)
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    n1, n2 = tree1.GetEntries(), tree2.GetEntries()
    max_input_events = max(n1, n2)

    fout = ROOT.TFile.Open(output_file, "RECREATE")
    fout.SetCompressionAlgorithm(ROOT.kZLIB)
    fout.SetCompressionLevel(1)
    out_tree = ROOT.TTree("tree", "Merged alternating events")

    out_branches = {}
    reader1 = ROOT.TTreeReader(tree1)
    reader2 = ROOT.TTreeReader(tree2)
    reader_vals1 = {}
    reader_vals2 = {}

    for branch in tree1.GetListOfBranches():
        name = branch.GetName()
        leaf = branch.GetLeaf(name)
        typename = leaf.GetTypeName()

        if typename == "Int_t":
            arr = array.array("i", [0])
            out_tree.Branch(name, arr, f"{name}/I")
            out_branches[name] = ("scalar", arr)
            reader_vals1[name] = ROOT.TTreeReaderValue["Int_t"](reader1, name)
            reader_vals2[name] = ROOT.TTreeReaderValue["Int_t"](reader2, name)
        elif typename == "vector<int>":
            vec = ROOT.std.vector("int")()
            out_tree.Branch(name, vec)
            out_branches[name] = ("vector", vec)
            reader_vals1[name] = ROOT.TTreeReaderValue["vector<int>"](reader1, name)
            reader_vals2[name] = ROOT.TTreeReaderValue["vector<int>"](reader2, name)
        elif typename == "vector<double>":
            vec = ROOT.std.vector("double")()
            out_tree.Branch(name, vec)
            out_branches[name] = ("vector", vec)
            reader_vals1[name] = ROOT.TTreeReaderValue["vector<double>"](reader1, name)
            reader_vals2[name] = ROOT.TTreeReaderValue["vector<double>"](reader2, name)
        else:
            print(f"[WARN] Skipping unsupported type '{typename}' for branch '{name}'")

    # Always override eventID
    out_branches["eventID"] = ("scalar", array.array("i", [0]))
    out_tree.Branch("eventID", out_branches["eventID"][1], "eventID/I")

    def handle_interrupt(sig, frame):
        print("\n[INFO] Interrupted. Writing partial file.")
        fout.cd()
        out_tree.Write()
        fout.Close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_interrupt)

    def copy_entry(reader, reader_vals, eid):
        reader.Next()
        for name, (kind, container) in out_branches.items():
            if name == "eventID":
                container[0] = eid
                continue
            val = reader_vals[name].Get()
            if kind == "scalar":
                container[0] = val
            elif kind == "vector":
                container.clear()
                for x in val:
                    container.push_back(x)
        out_tree.Fill()

    eid = 0
    for i in range(max_input_events):
        if MAX_OUTPUT_EVENTS is not None and eid >= MAX_OUTPUT_EVENTS:
            break
        if i < n1 and (MAX_OUTPUT_EVENTS is None or eid < MAX_OUTPUT_EVENTS):
            reader1.SetEntry(i)
            copy_entry(reader1, reader_vals1, eid)
            eid += 1
        if i < n2 and (MAX_OUTPUT_EVENTS is None or eid < MAX_OUTPUT_EVENTS):
            reader2.SetEntry(i)
            copy_entry(reader2, reader_vals2, eid)
            eid += 1
        if eid % 10000 == 0 and eid > 0:
            print(f"Merged {eid} events...")

    fout.cd()
    out_tree.Write()
    fout.Close()
    print(f"[INFO] Final output: {eid} events written to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternate merge events from two ROOT files.")
    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)
    parser.add_argument("--output", type=str, default="single_muons.root")
    args = parser.parse_args()

    merge_alternating_reader(args.file1, args.file2, args.output)

