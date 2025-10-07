import ROOT
import argparse


def skim_root_file(args):
    # Open input file
    fin = ROOT.TFile.Open(args.input_file, "READ")
    if not fin or fin.IsZombie():
        raise IOError(f"Could not open file: {args.input_file}")

    # Get input tree
    tree = fin.Get("tree")
    if not tree:
        raise KeyError("TTree 'tree' not found in the input file.")

    # Open output file
    fout = ROOT.TFile.Open(args.output_file, "RECREATE", "", ROOT.kZSTD)
    fout.SetCompressionLevel(3)

    # Copy entries
    n_entries = min(tree.GetEntries() - args.start, args.max_events)
    skimmed_tree = tree.CopyTree("", "", n_entries, args.start)

    # Write output
    skimmed_tree.Write()
    fout.Close()
    fin.Close()

    print(f"Skimmed {n_entries} events to '{args.output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skim a ROOT file to keep only the first N events.")
    parser.add_argument("input_file", type=str, help="Input ROOT file")
    parser.add_argument("--output_file", type=str, default="skimmed_output.root", help="Output ROOT file")
    parser.add_argument("--max_events", type=int, default=2000, help="Max events to keep")
    parser.add_argument("--start", type=int, default=0, help="Event to start skimming.")

    args = parser.parse_args()

    skim_root_file(args)
