import ROOT
import argparse
import random


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
    if args.random:
        # Randomly sample N events from input
        rng = random.Random(args.random_seed)

        n_total = tree.GetEntries()
        n_entries = min(n_total, args.max_events)

        indices = list(range(0, n_total))
        indices = rng.sample(indices, n_entries)
        indices.sort()

        skimmed_tree = tree.CloneTree(0)
        for i in indices:
            tree.GetEntry(i)
            skimmed_tree.Fill()
    else:
        # Just take the first N events from start
        n_entries = min(tree.GetEntries() - args.start, args.max_events)
        skimmed_tree = tree.CopyTree("", "", n_entries, args.start)

    # Write output
    skimmed_tree.Write()
    fout.Close()
    fin.Close()

    print(f"Skimmed {n_entries} events to '{args.output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skim a ROOT file to keep only the first N events."
    )
    parser.add_argument("input_file", type=str, help="Input ROOT file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="skimmed_output.root",
        help="Output ROOT file",
    )
    parser.add_argument(
        "--max_events", type=int, default=2000, help="Max events to keep"
    )
    parser.add_argument("--start", type=int, default=0, help="Event to start skimming.")
    parser.add_argument("--random", type=int, default=0, help="Whether to randomly sample events (1 = yes, 0 = no).")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    skim_root_file(args)
