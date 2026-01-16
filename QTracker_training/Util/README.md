

## ROOT Utility Scripts for Dimuon and Detector Analysis

This repository provides a suite of Python-based tools to explore, visualize, and process ROOT files commonly used in high-energy and nuclear physics
analyses. Each script is specialized for a task such as file structure inspection, event skimming, plotting invariant masses, and merging or visualizing
detector hit matrices.

## Requirements

- Python 3.x
- [ROOT](https://root.cern/)
- [uproot](https://github.com/scikit-hep/uproot)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

Make sure that ROOT is installed and properly sourced in your environment.

---

## Scripts Overview

### `file_structure.py`

**Purpose:**  
Prints the structure of a ROOT file, including TTrees and their branch layouts.

**Usage:**  
```bash
python3 file_structure.py file.root
```

This will print:
- The file-level structure (via `ls()`)
- The structure of all `TTree` objects within the file (`Print()`)

---

### `skim.py`

**Purpose:**  
Creates a skimmed version of a ROOT file, copying only the first `N` events from the TTree named `"tree"`.

**Edit These Variables to Customize:**
```python
OUTPUT_FILE = "skimmed_output.root"
NUM_EVENTS_TO_KEEP = 1000
```

**Usage:**  
```bash
python3 skim.py input_file.root
```

This creates a new ROOT file with the first `NUM_EVENTS_TO_KEEP` events.

---

### `imass_plot.py`

**Purpose:**  
Calculates and plots the invariant mass of muon pairs (mu+ and mu−) using their reconstructed momenta from a ROOT file.

**Expected branches:**
- `muID` (1 for mu+, 2 for mu−)
- `qpx`, `qpy`, `qpz` (momentum components)

**Usage:**  
```bash
python3 imass_plot.py file.root --output_plot mass_plot.png
```

**Output:**  
- Histogram of invariant mass
- PNG image file of the plot

---

### `merge_rus.py`

**Purpose:**  
Merges multiple ROOT files containing TTrees named `"tree"` into a single file named `merged_RUS.root`.

**Usage:**  
```bash
python3 merge_rus.py file1.root file2.root ...
```

Supports shell wildcards:
```bash
python3 merge_rus.py data/run*.root
```

**Output:**  
- Combined ROOT file with all entries from all files

---

### `plot_HitMatrix.py`

**Purpose:**  
Plots the 2D detector hit matrix (`elementID` vs `detectorID`) for a given event in a ROOT file. Ideal for visualizing detector occupancy or hit patterns.

**Usage:**  
```bash
python3 plot_HitMatrix.py yourfile.root -event 42
```

**Expected branches:**
- `detectorID` and `elementID` stored as vector branches

**Output:**  
- A graphical window displaying a heatmap of detector hits

---

### `plot_matrix.py`

**Purpose:**  
Plots the 2D detector hit matrix (`elementID` vs `detectorID`) for the output of QTracker_prod assuming the flag for declustering was used and the TMatrixD were saved to the output. This is for checking the hit matrix before and after declustering.

**Usage:**  
```bash
python3 plot_matrix.py yourfile.root -event 42
```

**Expected branches:**
- `detectorID` and `elementID` stored as vector branches

**Output:**  
- A graphical window displaying a heatmap of detector hits

---

### `plot_smax.py`

**Purpose:**  
Plots the 2D detector softmax matrix (`elementID` vs `detectorID`) for the output of QTracker_prod assuming the flag USE_SMAXMATRIX = True is set so that the softmax probability from the track finding CNN is saved storing the softmax probability for mu+ and mu- in separate TMatrixD matricies.

**Usage:**  
```bash
python3 plot_smax.py yourfile.root -event 42
```

**Expected branches:**
- `detectorID` and `elementID` stored as vector branches

**Output:**  
- A graphical window displaying a heatmap of detector hits

---


## File Structure

```
├── file_structure.py      # Inspect structure of ROOT files
├── skim.py                # Make skimmed files with reduced events
├── imass_plot.py          # Plot invariant mass from muon momenta
├── merge_rus.py           # Merge multiple RUS-format ROOT files
├── plot_HitMatrix.py      # Visualize hit matrix for specific events
```

---

## Notes

- All scripts assume the TTree is named `tree`. Modify the code if your ROOT files use a different name.
- These utilities are especially useful for developing the needed pipelines or checking data quality in high-multiplicity events like dimuon final states or spectrometer hits.
