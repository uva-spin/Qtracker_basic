QTracker_prod is a set of learning scripts that is more closely representative of the real production version.  Start with QTracker_basic and learn
what its doing, then explore the data generation and training here.  QTracker_prod.py used the most updated version of the RUS file while QTracker_basic uses only a limited version with limited variables.  QTracker_basic is meant to be very simple just for getting started and will not be updated for further data features and improved handling. QTracker_prod will be updated as the RUS file framework evolves.


# RUS Track and Momentum Reconstruction with QTracker_prod

This repository contains a Python-based reconstruction pipeline designed to process RUS files. The script reads raw detector hit information, reconstructs the muon tracks, predicts the muon momenta, and outputs the results to a new ROOT file, enabling further physics analysis.

## Overview

The reconstruction process includes:
- Loading detector hit information (detectorID, elementID, driftDistance, tdcTime) from a ROOT file.
- (Optional) Declusterizing noisy or clustered hits.
- Predicting hit arrays for muon tracks (`μ⁺` and `μ⁻`) using a trained Deep Neural Network (TrackFinder).
- Refining the predicted hit arrays by matching them to real recorded detector hits.
- Predicting the momentum components (`pₓ`, `pᵧ`, `p_z`) of each muon using separate DNN models for `μ⁺` and `μ⁻`.
- (Optional) Predicting a track quality metric (χ²) using a dedicated DNN.
- Writing reconstructed information, including hit arrays, drift distances, momentum vectors, and χ² values, into a new compressed ROOT file.

Optional debugging outputs:
- Hit matrices before and after declustering.
- Softmax output matrices from the TrackFinder model.

---

## Installation

This reconstruction pipeline depends on the following packages:
- [ROOT](https://root.cern/)
- [NumPy](https://numpy.org/)
- [TensorFlow 2.x](https://www.tensorflow.org/)
- [Numba](https://numba.pydata.org/)

You can install the Python dependencies with:

```bash
pip install numpy tensorflow numba
```

Ensure that your ROOT installation includes PyROOT support.

---

## Usage

Run the reconstruction script with:

```bash
python3 QTracker_prod.py path/to/input.root --output_file path/to/output.root
```

Arguments:
- `root_file` (required): Path to the input ROOT file containing detector hit information.
- `--output_file` (optional): Path to the output ROOT file. Defaults to `qtracker_reco.root`.

---

## Model Files

This script expects pre-trained TensorFlow `.h5` model files stored in the `./models/` directory:

| Purpose | Model Path | Description |
|:---|:---|:---|
| Track Finder | `models/track_finder.h5` | Predicts the hit patterns of muon tracks. |
| Momentum Predictor (μ⁺) | `models/mom_mup.h5` | Predicts momentum `(pₓ, pᵧ, p_z)` for positive muons. |
| Momentum Predictor (μ⁻) | `models/mom_mum.h5` | Predicts momentum `(pₓ, pᵧ, p_z)` for negative muons. |
| χ² Metric Model | `models/chi2_predictor_model.h5` | (Optional) Predicts a track quality χ² metric. |

> If you are running the reconstruction for the **first time** to prepare data for training the χ² model, you must **set `USE_CHI2 = False`** in the script.

---

## Main Features

- **Declustering**  
  Noise and multi-hit clusters in the detector can optionally be cleaned using a parallelized Numba declustering function.

- **Track Finding**  
  Uses a Deep Neural Network to predict the detector element hit arrays separately for `μ⁺` and `μ⁻`.

- **Momentum Inference**  
  Predicts three-momentum vectors for each track using dedicated DNN models.

- **Track Quality Prediction (χ² Metric)**  
  If enabled, a separate model predicts a track quality figure of merit for each reconstructed track.

- **Flexible Output**  
  The output ROOT file includes:
  - Reconstructed hit arrays (`qHitArray_mup`, `qHitArray_mum`)
  - Reconstructed drift distances
  - Reconstructed momentum components (`qpx`, `qpy`, `qpz`)
  - (Optional) Predicted χ² values (`qchi2`)
  - (Optional) Hit matrices before and after declustering
  - (Optional) Softmax response matrices

---

## Important Settings

Modify the following settings at the top of `QTracker_prod.py` to control behavior:

```python
USE_CHI2 = True          # Set to False if generating data for chi2 model training.
USE_DECLUSTERING = False # Set to True to clean hit matrices using declustering.
USE_SMAXMATRIX = False   # Set to True to write softmax outputs to ROOT.
```

---

## Output

The output file (default `qtracker_reco.root`) will contain all original event information along with new branches:
- `qHitArray_mup` and `qHitArray_mum` — reconstructed elementIDs for each detector.
- `driftDistance_mup` and `driftDistance_mum` — drift distances associated with each hit.
- `qpx`, `qpy`, `qpz` — reconstructed momentum components for each track.
- `qchi2` — (optional) predicted χ² metric.

Optional auxiliary trees:
- `hitMatrixTree` — hit matrices before and after declustering.
- `softmaxTree` — softmax prediction matrices from the TrackFinder model.

---

## Notes

- The TrackFinder model requires a custom loss function (`custom_loss`) for loading.
- Track momentum predictions mask out unused detector stations automatically.
- If you wish to generate new training data for the χ² model, you must first run the script with `USE_CHI2 = False`.

---

## Example

```bash
python3 QTracker_prod.py input_data.root --output_file reco_output.root
```



