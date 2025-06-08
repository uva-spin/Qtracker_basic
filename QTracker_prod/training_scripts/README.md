# QTracker: Model Training

## Overview
QTracker is a framework for reconstructing and analyzing muon tracks in particle physics experiments. This repository provides scripts for preparing the Monte Carlo generated events and training models for track finding and momentum reconstruction, and evaluating reconstructed tracks using a quality metric.

## Prerequisites
At this stage of QTracker (beta) production, we train the Track Finder and Momentum models separately so that they can be evaluated independently, with no interdependence. Once the Track Finder is optimized for a particular occupancy, we must train the Momentum model using the output from the Track Finder. For now, however, the procedure is simple: the Track Finder handles all the background, and the Momentum models assume (approximately) correct HitArrays with negligible error.

The goal of the Track Finder is to evaluate the hits in each event and assign a probability to each detector element indicating the likelihood that it is associated with a reconstructable dimuon track originating from the target.

## Data Preparation
1. **Splitting Signal Data into Mu+ and Mu- Tracks**
   
   Use `separate.py` to split J/psi, Drell-Yan (DY), or two-muon tracks (mu+/mu-) into separate files:
   ```sh
   python3 QTracker_prod/data/separate.py JPsi_Target.root
   ```
   This will generate:
   - `JPsi_Target_track1.root` (mu+ tracks)
   - `JPsi_Target_track2.root` (mu- tracks)

2. **Generating Hit Arrays for Training**
   
   The `gen_training.py` script processes the separated muon tracks and prepares the necessary hit arrays for model training:
   ```sh
   python3 QTracker_prob/data/gen_training.py JPsi_Target_track1.root JPsi_Target_track2.root
   ```
   This will produce the following training data files:
   - `finder_training.root` (for track finding training)
   - `momentum_training-1.root` (for mu+ momentum training)
   - `momentum_training-2.root` (for mu- momentum training)

## Model Training

### 1. Training the Track Finder
To perform a training test with no background you can train with the clean output of the pure dimuons from the target with no injected background or noise.
```sh
python3 QTracker_prob/training_scripts/TrackFinder_train.py finder_training.root
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
python3 QTracker_prod/training_scripts/QTracker_basic.py JPsi_Dump.root
```
This will generate:
- `qtracker_reco.root` (Reconstructed output file)

## Evaluating Reconstruction Quality
### 1. Checking the Invariant Mass Spectrum
```sh
python3 QTracker_prod/training_scripts/imass_plot.py qtracker_reco.root
```
This script will plot the mass spectrum of your reconstructed events.

![invariant_mass](https://github.com/user-attachments/assets/8654506c-ce7c-4458-933b-6d117029bf60)

When everything is working correctly you should see a J/psi mass peak assuming you training using the J/psi Monte Carlo file.
This is a good confirmation everything is working and you are ready to add in noise and more complicated backgrounds.

### 2. Training the Quality Metric Model (Chi-Squared Method)
```sh
python3 QTracker_prod/training_scripts/Qmetric_training.py qtracker_reco.root
```

## Notes
- Ensure that your dataset follows the expected RUS format before processing.
- The trained models should be stored in the correct directory (`QTracker_prod/models`) for proper operation.
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





# Updating The Dimuon Track Finder Suite

This repository contains three progressively advanced deep learning models for identifying true dimuon tracks from noisy hit arrays in a high-energy
and nuclear physics detector at SpinQuest. Each script shares a common goal—distinguishing the complete muon+ and muon− trajectories—but uses increasingly
sophisticated architectures to improve accuracy, robustness, and generalization.  The goal is to be able to determine the good reconstructable tracks from 
the target dimuons no matter the level of background hits (mostly from the beam dump) in the event.  The background hit patterns, occupancy, track completeness
per station and per detector must be simulated well in the training data.  Detailed studies of the quality of extraction are required along with detailed
measurememts of the uncertainties imposed by various background characteristics.

All models use RUS ROOT TTree and are trained to determine the hits and fill a Hit Array with the hits from the true dimuon tracks.

---

## Scripts Overview

### `TrackFinder_prod.py` – Baseline VGG-Style CNN

**Overview:**  
This is the production-ready baseline model. It uses a deep VGG-style Convolutional Neural Network (CNN) to classify and reconstruct dimuon tracks from 2D binary hit arrays.

**Architecture Highlights:**
- 5 convolutional blocks (VGG16-style)
- Fully connected layers with dropout
- Predicts both mu+ and mu− hit arrays simultaneously
- Includes an overlap penalty in the loss function to discourage track confusion

**Advantages:**
- Solid baseline accuracy
- Easy to train and interpret
- Good general performance on clean data

---

### `TrackFinder_acc.py` – VGG + Residual Blocks (ResNet Hybrid)

**Overview:**  
This model introduces residual connections for deeper and more stable learning. It retains a VGG-style input stage but adds skip-connected residual blocks for better gradient flow and learning of complex features.

**Architecture Highlights:**
- VGG frontend followed by 4 stacked residual blocks
- L2 regularization in convolutional layers
- Better optimization for deep networks
- Same custom loss function with overlap penalty

**Advantages Over Baseline:**
- Improved convergence speed and generalization
- Better performance in dense or ambiguous backgrounds
- More resilient to overfitting due to regularization and skip connections

---

### `TrackFinder_attention.py` – ResNet + CBAM (Attention Mechanisms)

**Overview:**  
The most advanced version of the model combines the residual ResNet blocks with CBAM (Convolutional Block Attention Module), enabling the network to focus attention dynamically on the most informative regions of the input space.

**Architecture Highlights:**
- Same residual backbone as `TrackFinder_acc.py`
- Channel and spatial attention modules after each residual block
- Attention helps model prioritize true tracks and suppress noise
- Same custom loss function with enhanced output focus

**Advantages Over ResNet Hybrid:**
- Further boosts precision in high-noise environments
- Enhanced interpretability via attention maps
- Outperforms others on challenging test sets with partial tracks and background confusion

---

## Input Format

All scripts expect a ROOT file containing a `TTree` named `"tree"` with the following vector branches:

- `detectorID`, `elementID` – Coordinates of all hits in the event  
- `HitArray_mup`, `HitArray_mum` – Ground-truth hit indices for the two muon tracks (to be learned)

Hit arrays are represented as 2D binary matrices of shape `(62, 201)`.

---

## Usage

### Train a model
Use the tools in the data folder to make messy events in the RUS file formate for training and testing.
```bash
python TrackFinder_prod.py mc_messy.root --output_model=models/track_finder.h5
```

Swap in `TrackFinder_acc.py` or `TrackFinder_attention.py` to train the ResNet or CBAM model.

Optional arguments:
- `--learning_rate` – Adjust the learning rate (default: `5e-5`)
- `--output_model` – Change output path for the trained model

---

## Loss Function

All models use a **custom loss** combining:
- Sparse categorical cross-entropy (for each muon)
- A penalty on the predicted overlap between mu+ and mu− tracks

This encourages the network to separate the two tracks cleanly while matching ground truth.

---

## Output

After training, a Keras `.h5` model file is saved to the `models/` directory. You can later use this model for inference on test RUS files.

---


## Requirements

- Python 3.9+
- ROOT (PyROOT interface)
- TensorFlow ≥ 2.9
- NumPy, scikit-learn

---
