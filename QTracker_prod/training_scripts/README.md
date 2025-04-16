
# Dimuon Track Finder Suite

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
