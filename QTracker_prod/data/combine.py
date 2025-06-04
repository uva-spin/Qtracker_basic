import ROOT
import argparse
import signal
import sys
import array

# Set this to limit total events written. Set to None to go until one file ends.
MAX_OUTPUT_EVENTS = 100000

def merge_alternating_reader(file1, file2, output_file):
    f1 = ROOT.TFile.Open(file1)
    f2 = ROOT.TFile.Open(file2)
    tree1 = f1.Get("tree")
    tree2 = f2.Get("tree")

    n1, n2 = tree1.GetEntries(), tree2.GetEntries()
    max_input_events = max(n1, n2)

    fout = ROOT.TFile.Open(output_file, "RECREATE")
    out_tree = ROOT.TTree("tree", "Merged alternating events")
    # Disable auto-save to prevent intermediate writes
    out_tree.SetAutoSave(0)  # Disable auto-save (0 or negative value)

    out_branches = {}
    reader1 = ROOT.TTreeReader(tree1)
    reader2 = ROOT.TTreeReader(tree2)
    reader_vals1 = {}
    reader_vals2 = {}

    for branch in tree1.GetListOfBranches():
        name = branch.GetName()
        leaf = branch.GetLeaf(name)
        typename = leaf.GetTypeName()
        is_array = leaf.GetLen() > 1 or leaf.GetLeafCount() is not None

        if name == "eventID":  # Skip eventID, handle it separately
            continue

        try:
            if typename == "Int_t" and not is_array:
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
            elif is_array:
                # Handle arrays (e.g., Float_t[10], Double_t[10])
                if typename in ["Float_t", "Double_t"]:
                    arr_type = "f" if typename == "Float_t" else "d"
                    arr = array.array(arr_type, [0] * leaf.GetLen())
                    out_tree.Branch(name, arr, f"{name}[{leaf.GetLen()}]/{arr_type.upper()}")
                    out_branches[name] = ("array", arr)
                    reader_vals1[name] = ROOT.TTreeReaderArray[typename](reader1, name)
                    reader_vals2[name] = ROOT.TTreeReaderArray[typename](reader2, name)
                else:
                    print(f"[WARN] Skipping unsupported array type '{typename}' for branch '{name}'")
            else:
                print(f"[WARN] Skipping unsupported type '{typename}' for branch '{name}'")
        except Exception as e:
            print(f"[ERROR] Failed to process branch '{name}' (type: {typename}): {str(e)}")
            continue

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
            val = reader_vals[name].Get() if kind != "array" else reader_vals[name]
            if kind == "scalar":
                container[0] = val
            elif kind == "vector":
                container.clear()
                for x in val:
                    container.push_back(x)
            elif kind == "array":
                for i in range(min(len(val), len(container))):
                    container[i] = val[i]
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
            print(f"[INFO] Merged {eid} events...")

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