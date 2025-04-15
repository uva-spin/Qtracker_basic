import ROOT
import argparse


OUTPUT_FILE = "skimmed_output.root"     # Output ROOT file
NUM_EVENTS_TO_KEEP = 50                 # Number of events to keep


def skim_root_file(input_file, output_file, max_events):
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

    # Copy entries
    n_entries = min(tree.GetEntries(), max_events)
    for i in range(n_entries):
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

    args = parser.parse_args()
    skim_root_file(args.input_file, OUTPUT_FILE, NUM_EVENTS_TO_KEEP)
