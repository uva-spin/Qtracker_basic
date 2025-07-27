import ROOT
import argparse


def skim_root_file(input_file, output_file, max_events, start=0):
    # Open input file
    fin = ROOT.TFile.Open(input_file, "READ")
    if not fin or fin.IsZombie():
        raise IOError(f"Could not open file: {input_file}")

    # Get input tree
    tree = fin.Get("tree")
    if not tree:
        raise KeyError("TTree 'tree' not found in the input file.")

    # Open output file
    fout = ROOT.TFile.Open(output_file, "RECREATE", "", ROOT.kLZMA)
    fout.SetCompressionLevel(9)

    # Clone tree structure (empty)
    skimmed_tree = tree.CloneTree(0)
    skimmed_tree.SetAutoSave(0)

    # Copy entries
    n_entries = min(tree.GetEntries() - start, max_events)
    end = start + n_entries
    for i in range(start, end):
        tree.GetEntry(i)
        skimmed_tree.Fill()

    # Write output
    fout.Write()
    fout.Close()
    fin.Close()

    print(f"Skimmed {n_entries} events to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skim a ROOT file to keep only the first N events.")
    parser.add_argument("input_file", type=str, help="Input ROOT file")
    parser.add_argument("--output_file", type=str, default="skimmed_output.root", help="Output ROOT file")
    parser.add_argument("--max_events", type=int, default=2000, help="Max events to keep")
    parser.add_argument("--start", type=int, default=0, help="Event to start skimming.")

    args = parser.parse_args()

    skim_root_file(args.input_file, args.output_file, args.max_events, args.start)
