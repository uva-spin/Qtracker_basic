# QTracker: Muon Track Reconstruction and Momentum Training

## Overview
QTracker is a framework for reconstructing and analyzing muon tracks in particle physics experiments. This repository provides scripts for processing raw data, training models for track finding and momentum reconstruction, and evaluating reconstructed tracks using a quality metric.

## Prerequisites
Ensure that your dataset can be reliably organized into single-track data with identified mu+ and mu- events.

## Data Preparation
1. **Splitting Data into Mu+ and Mu- Tracks**
   
   Use `separate.py` to split J/psi, Drell-Yan (DY), or two-muon tracks (mu+/mu-) into separate files:
   ```sh
   python3 QTracker_basic/data/separate.py JPsi_Dump.root
   ```
   This will generate:
   - `JPsi_Dump_track1.root` (mu+ tracks)
   - `JPsi_Dump_track2.root` (mu- tracks)

2. **Generating Hit Arrays for Training**
   
   The `gen_training.py` script processes the separated muon tracks and prepares the necessary hit arrays for model training:
   ```sh
   python3 QTracker_basic/data/gen_training.py JPsi_Dump_track1.root JPsi_Dump_track2.root
   ```
   This will produce the following training data files:
   - `finder_training.root` (for track finding training)
   - `momentum_training-1.root` (for mu+ momentum training)
   - `momentum_training-2.root` (for mu- momentum training)

## Model Training

### 1. Training the Track Finder
```sh
python3 QTracker_basic/training_scripts/TrackFinder_train.py finder_training.root
```

### 2. Training the Momentum Reconstruction Models
```sh
python3 QTracker_basic/training_scripts/Momentum_training.py --output mom_mup.h5 momentum_training-1.root
python3 QTracker_basic/training_scripts/Momentum_training.py --output mom_mum.h5 momentum_training-2.root
```

Store the resulting models in the `QTracker_basic/models` directory.

## Testing the Tracker
To test the trained models on a dataset:
```sh
python3 QTracker_basic/training_scripts/QTracker_basic.py JPsi_Dump.root
```
This will generate:
- `qtracker_reco.root` (Reconstructed output file)

## Evaluating Reconstruction Quality
### 1. Checking the Invariant Mass Spectrum
```sh
python3 QTracker_basic/training_scripts/imass_plot.py qtracker_reco.root
```
This script will plot the mass spectrum of your reconstructed events.

### 2. Training the Quality Metric Model (Chi-Squared Method)
```sh
python3 QTracker_basic/training_scripts/Qmetric_training.py qtracker_reco.root
```

## Notes
- Ensure that your dataset follows the expected RUS format before processing.
- The trained models should be stored in the correct directory (`QTracker_basic/models`) for proper operation.
- The scripts assume that dependencies such as ROOT, Python, and required ML libraries are properly installed.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy tensorflow uproot sklearn
```
Additionally, you need `ROOT` installed to process ROOT files.

## Scripts Overview

### 1. `Trackfinder_training.py`
This script trains a Convolutional Neural Network (CNN) model to predict hit arrays from detector hit matrices.

#### Usage:
```bash
python QTracker_basic/training_scripts/TrackFinder_training.py <root_file> --output_model models/track_finder.h5
```

#### Functionality:
- Loads hit data from a ROOT file, converting it into a binary hit matrix.
- Uses a CNN model with dropout and batch normalization to reduce overfitting.
- Predicts the hit arrays for mu+ and mu- particles.
- Saves the trained model.

---

### 2. `Momentum_training.py`
This script trains a deep neural network (DNN) to predict momentum components (gpx, gpy, gpz) from detector hit arrays.

#### Usage:
```bash
python QTracker_basic/training_scripts/Momentum_training.py <input_root_files> --output models/mom_model.h5
```

#### Functionality:
- Loads hit arrays and corresponding momentum components from multiple ROOT files.
- Applies preprocessing and normalization to input data.
- Uses a fully connected neural network with batch normalization and dropout.
- Saves the trained model for later use.

---

### 3. `Qmetric_training.py`
This script trains a model to predict chi-squared (χ²) values based on reconstructed and true momentum components.

#### Usage:
```bash
python QTracker_basic/training_scripts/Qmetric_training.py <root_file>
```

#### Functionality:
- Extracts reconstructed and true momentum values from a ROOT file.
- Computes the χ² value for each event based on momentum differences.
- Uses a fully connected neural network with dropout and L2 regularization.
- Trains the model and saves it for later use.

## Model Outputs
Each script saves trained models in the `models/` directory:
- `track_finder.h5` - CNN model predicting detector hit arrays.
- `mom_model.h5` - DNN model predicting momentum components.
- `chi2_predictor_model.h5` - Model predicting χ² values for track assessment.




