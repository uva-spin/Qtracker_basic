import ROOT
import argparse
import random

# =========================
# Split configuration
# =========================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

SEED = 42
# =========================


def split_root_file(args):
    # Open input file
    fin = ROOT.TFile.Open(args.input_file, "READ")
    if not fin or fin.IsZombie():
        raise IOError(f"Could not open file: {args.input_file}")

    # Get input tree
    tree = fin.Get("tree")
    if not tree:
        raise KeyError("TTree 'tree' not found in the input file.")

    n_entries = tree.GetEntries()
    print(f"Total events in input: {n_entries}")

    # Sanity check on ratios
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1")

    # Generate shuffled indices
    indices = list(range(n_entries))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    n_train = int(TRAIN_RATIO * n_entries)
    n_val   = int(VAL_RATIO * n_entries)
    n_test  = n_entries - n_train - n_val

    labels = [0] * n_entries  # 0=train, 1=val, 2=test

    for i in indices[:n_train]:
        labels[i] = 0
    for i in indices[n_train:n_train + n_val]:
        labels[i] = 1
    for i in indices[n_train + n_val:]:
        labels[i] = 2

    # Open output files
    fout_train = ROOT.TFile.Open(args.train_output, "RECREATE", "", ROOT.kZSTD)
    fout_val   = ROOT.TFile.Open(args.val_output,   "RECREATE", "", ROOT.kZSTD)
    fout_test  = ROOT.TFile.Open(args.test_output,  "RECREATE", "", ROOT.kZSTD)

    for f in (fout_train, fout_val, fout_test):
        f.SetCompressionLevel(3)

    # Clone trees (structure only)
    fout_train.cd()
    train_tree = tree.CloneTree(0)

    fout_val.cd()
    val_tree = tree.CloneTree(0)

    fout_test.cd()
    test_tree = tree.CloneTree(0)

    # Fill trees
    for i in range(n_entries):
        tree.GetEntry(i)
        if labels[i] == 0:
            train_tree.Fill()
        elif labels[i] == 1:
            val_tree.Fill()
        else:
            test_tree.Fill()

        if i > 0 and i % 1_000_000 == 0:
            print(f"Processed {i}/{n_entries}")

    # Write output
    fout_train.cd()
    train_tree.Write()

    fout_val.cd()
    val_tree.Write()

    fout_test.cd()
    test_tree.Write()

    fout_train.Close()
    fout_val.Close()
    fout_test.Close()
    fin.Close()

    print("=== DONE ===")
    print(f"Wrote train file: {args.train_output}")
    print(f"Wrote val file:   {args.val_output}")
    print(f"Wrote test file:  {args.test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a ROOT TTree into train/val/test datasets."
    )
    parser.add_argument("input_file", type=str, help="Input ROOT file")

    parser.add_argument(
        "--train_output",
        type=str,
        default="train.root",
        help="Output ROOT file for training set",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val.root",
        help="Output ROOT file for validation set",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="test.root",
        help="Output ROOT file for test set",
    )

    args = parser.parse_args()
    split_root_file(args)
