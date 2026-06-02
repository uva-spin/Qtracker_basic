import ROOT
import sys

# Check for correct usage
if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} file.root")
    sys.exit(1)

# Get the file path from the command line argument
file_path = sys.argv[1]

# Open the ROOT file
root_file = ROOT.TFile.Open(file_path)
if not root_file or root_file.IsZombie():
    print(f"Error: Could not open file '{file_path}'")
    sys.exit(1)

# Print the structure of the file using .ls()
print("ROOT file structure:")
root_file.ls()

# Optionally, list the branches and leaves for TTrees
def print_tree_structure(obj, name):
    if obj.ClassName() == "TTree":
        print(f"\nTTree '{name}' structure:")
        obj.Print()

for key in root_file.GetListOfKeys():
    obj = key.ReadObj()
    print_tree_structure(obj, key.GetName())

# Close the file
root_file.Close()
