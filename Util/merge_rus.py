import ROOT
import sys
import glob

# Check if at least one input file is provided
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} file1.root file2.root ...")
    sys.exit(1)

# Get the list of input ROOT files
input_files = []
for arg in sys.argv[1:]:
    input_files.extend(glob.glob(arg))  # Expand wildcard expressions

if not input_files:
    print("Error: No valid ROOT files found.")
    sys.exit(1)

# Create the output ROOT file
output_file = ROOT.TFile("merged_RUS.root", "RECREATE")

# Create an empty TChain to merge trees
chain = ROOT.TChain("tree")

# Add all input files to the chain
for file in input_files:
    chain.Add(file)

# Clone the tree structure to the output file
output_file.cd()
merged_tree = chain.CloneTree(-1)

# Write the merged tree to the output file
merged_tree.Write()
output_file.Close()

print(f"Merged {len(input_files)} files into 'merged_RUS.root'.")
