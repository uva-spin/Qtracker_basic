# Qtracker Reconstruction Software: `Qtracker_basic.py`

`Qtracker_basic.py` is the main inference script in the **Qtracker reconstruction software**. It processes RUS ROOT files containing detector hit data, performs track reconstruction, predicts particle momenta, and writes the results along with the original data to a new RUS ROOT file. This script is a critical component of the Qtracker framework, enabling the reconstruction of particle tracks and the extraction of physical quantities such as momentum and chi-squared (χ²) values.  This version of Qtracker referred to as 'basic' is for learning purposes.  The models are overly simplistics but this make the full code light and easy to start experimenting with.

---

## Overview

The script performs the following steps:
1. **Loads Data**: Reads detector hit data from a ROOT file, including `detectorID`, `elementID`, and `driftDistance`.
2. **Track Finding**: Uses a pre-trained TensorFlow model to predict hit arrays for muon tracks (mu+ and mu-).
3. **Hit Array Refinement**: Refines the predicted hit arrays by matching them with actual hits from the input data.
4. **Momentum Prediction**: Predicts the momentum (px, py, pz) for each track using refined hit arrays.
5. **Chi-Squared Prediction**: Estimates the χ² values for the reconstructed tracks.
6. **Output**: Writes the refined hit arrays, predicted momenta, and χ² values to a new ROOT file, preserving the original data.

---

## Why Are There Two Data Loaders?

The script uses **two separate data loaders** to handle different aspects of the input ROOT file:

1. **First Data Loader (`load_data`)**:
   - **Purpose**: Loads the hit matrix for track finding.
   - **What It Does**:
     - Reads the `detectorID` and `elementID` branches from the ROOT file.
     - Constructs a binary hit matrix (`X`) of shape `(num_events, num_detectors, num_element_ids)`.
     - This matrix is used as input to the track-finding model to predict hit arrays for mu+ and mu- tracks.
   - **Why It Exists**:
     - The track-finding model requires a structured representation of hits (a binary matrix) to predict tracks efficiently.

2. **Second Data Loader (`load_detector_element_data`)**:
   - **Purpose**: Loads additional detector information for hit array refinement.
   - **What It Does**:
     - Reads the `detectorID`, `elementID`, and `driftDistance` branches from the ROOT file.
     - Stores these values as lists of arrays for each event.
   - **Why It Exists**:
     - The refinement step requires detailed information about the actual hits (e.g., `elementID` and `driftDistance`) to correct and refine the predicted hit arrays.
     - This ensures that the predicted hit arrays align with the actual detector data, improving the accuracy of the reconstruction.

By separating these tasks into two loaders, the script ensures modularity and efficiency. The first loader focuses on preparing data for the track-finding model, while the second loader provides the detailed information needed for refinement and momentum prediction.

---

## Requirements

- Python 3.8 or higher
- ROOT (PyROOT)
- TensorFlow 2.x
- NumPy

---

## Usage

1. **Install Dependencies**:
   Ensure you have the required libraries installed:
   ```bash
   pip install numpy tensorflow
   ```

2. **Run the Script**:
   Provide the input ROOT file and specify the output file (optional):
   ```bash
   python Qtracker_basic.py input_file.root --output_file output_file.root
   ```

   - `input_file.root`: Path to the input ROOT file containing detector hit data.
   - `--output_file`: (Optional) Path to the output ROOT file. Default is `qtracker_reco.root`.

---

## Models

The script uses the following pre-trained models:
- `track_finder.h5`: For track finding (predicting hit arrays).
- `mom_mup.h5`: For predicting momentum of mu+ tracks.
- `mom_mum.h5`: For predicting momentum of mu- tracks.
- `chi2_predictor_model.h5`: For predicting χ² values.

Ensure these models are located in the `./models/` directory or update the paths in the script.

---

## Output

The output ROOT file contains:
- Original data from the input ROOT file.
- Refined hit arrays for mu+ and mu- tracks.
- Predicted momenta (px, py, pz) for mu+ and mu- tracks.
- Predicted χ² values for the tracks.

---

## Customization

- **Model Paths**: Update the `MODEL_PATH_*` variables to point to your model files.
- **Detector Configuration**: Adjust `NUM_DETECTORS` and `NUM_ELEMENT_IDS` if your detector setup differs.
- **Loss Function**: Modify the `custom_loss` function if needed for your specific use case.

---

## Example

```bash
python3 Qtracker_basic.py input_data.root --output_file qtracker_reco.root
```

This will process `input_data.root` and save the results to `qtracker_reco.root`.

---

## Notes

- Ensure the input ROOT file contains a tree named `tree` with branches for `detectorID`, `elementID`, and `driftDistance`.
- The script assumes a specific detector configuration (62 detectors and 201 element IDs). Adjust these values if necessary.

