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
   python3 QTracker_training/data/separate.py JPsi_Target.root
   ```

   This will generate:

   - `JPsi_Target_track1.root` (mu+ tracks)
   - `JPsi_Target_track2.root` (mu- tracks)

2. **Generating Hit Arrays for Training**

   The `gen_training.py` script processes the separated muon tracks and prepares the necessary hit arrays for model training:

   ```sh
   python3 QTracker_training/data/gen_training.py JPsi_Target_track1.root JPsi_Target_track2.root
   ```

   This will produce the following training data files:

   - `finder_training.root` (for track finding training)
   - `momentum_training-1.root` (for mu+ momentum training)
   - `momentum_training-2.root` (for mu- momentum training)

## Model Training

### 1. Training the Track Finder

To perform a training test with no background you can train with the clean output of the pure dimuons from the target with no injected background or noise.

```sh
python3 TrackFinder.py mc_events_train.root mc_events_val.root
```

**To prepare data and train TrackFinder with background and noise, follow the procedure outlined [here](https://github.com/uva-spin/Qtracker_basic/tree/main/QTracker_training/data).**

### 2. Training the Momentum Reconstruction Models

```sh
python3 Momentum_training.py --output mom_mup.h5 momentum_training-1.root
python3 Momentum_training.py --output mom_mum.h5 momentum_training-2.root
```

Store the resulting models in the `QTracker_training/checkpoints` directory.

## Testing the Tracker

To test the trained models on a dataset:

```sh
python3 QTracker.py mc_events_val.root
```

This will generate:

- `qtracker_reco.root` (Reconstructed output file)

## Evaluating Reconstruction Quality

### 1. Checking the Invariant Mass Spectrum

```sh
python3 Util/imass_plot.py qtracker_reco.root
```

This script will plot the mass spectrum of your reconstructed events.

![invariant_mass](https://github.com/user-attachments/assets/8654506c-ce7c-4458-933b-6d117029bf60)

When everything is working correctly you should see a J/psi mass peak assuming you training using the J/psi Monte Carlo file.
This is a good confirmation everything is working and you are ready to add in noise and more complicated backgrounds.

### 2. Training the Quality Metric Model (Chi-Squared Method)

```sh
python3 Qmetric_training.py qtracker_reco.root
```

## Notes

- Ensure that your dataset follows the expected RUS format before processing.
- The trained models should be stored in the correct directory (`QTracker_training/checkpoints`) for proper operation.
- The scripts assume that dependencies such as ROOT, Python, and required ML libraries are properly installed.

## Requirements

Ensure you have the following dependencies installed:

```bash
conda install -c conda-forge numpy tensorflow uproot sklearn ROOT
```

## Scripts Overview

### 1. `TrackFinder.py`

This script trains a custom U-Net++ model with axial attention to predict hit arrays from detector hit matrices.

#### Usage:

```bash
python TrackFinder.py <root_file> --output_model checkpoints/track_finder.keras
```

#### Functionality:

- Loads hit data from a ROOT file, converting it into a binary hit matrix.
- Uses a U-Net++ denoiser and segmenter pipeline to reconstruct one track in an event with variable noise.
- Predicts the hit arrays for mu+ and mu- particles.
- Saves the trained model.

---

### 2. `Momentum_training.py`

This script trains a deep neural network (DNN) to predict momentum components (gpx, gpy, gpz) from detector hit arrays.

#### Usage:

```bash
python Momentum_training.py <input_root_files> --output checkpoints/mom_model.h5
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
python Qmetric_training.py <root_file>
```

#### Functionality:

- Extracts reconstructed and true momentum values from a ROOT file.
- Computes the χ² value for each event based on momentum differences.
- Uses a fully connected neural network with dropout and L2 regularization.
- Trains the model and saves it for later use.

## Model Outputs

Each script saves trained models in the `checkpoints/` directory:

- `track_finder.keras` - CNN model predicting detector hit arrays.
- `mom_model.h5` - DNN model predicting momentum components.
- `chi2_predictor_model.h5` - Model predicting χ² values for track assessment.

# Updating The Dimuon Track Finder Suite

This repository contains three progressively advanced deep learning models for identifying true dimuon tracks from noisy hit arrays in a high-energy
and nuclear physics detector at SpinQuest. Each script shares a common goal—distinguishing the complete muon+ and muon− trajectories—but uses increasingly
sophisticated architectures to improve accuracy, robustness, and generalization. The goal is to be able to determine the good reconstructable tracks from
the target dimuons no matter the level of background hits (mostly from the beam dump) in the event. The background hit patterns, occupancy, track completeness
per station and per detector must be simulated well in the training data. Detailed studies of the quality of extraction are required along with detailed
measurememts of the uncertainties imposed by various background characteristics.

All models use RUS ROOT TTree and are trained to determine the hits and fill a Hit Array with the hits from the true dimuon tracks.

---

## Scripts Overview

### `TrackFinder.py` – U-Net++ denoiser and segmenter pipeline

**Overview:**  
This is the production-ready model for detecting a single particle track in a given event.

**Architecture Highlights:**

- U-Net++ based denoiser that removes background tracks from a given event.
- U-Net++ with axial attention based segmenter that extracts a ground-truth track from the denoised event.
- Denoiser-segmenter pipeline trained end-to-end.

**Advantages:**

- Solid baseline accuracy (up to 93% validation accuracy on noisy events)
- Easy to train and interpret
- Good general performance on clean data

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
python TrackFinder.py mc_events_train.root mc_events_val.root --output_model=checkpoints/track_finder.keras
```

Optional arguments:

### Command-line arguments

- `train_root_file_low`  
  Path to the low complexity training ROOT file (required).

- `val_root_file`  
  Path to the validation ROOT file (required).

- `--train_root_file_med`  
  Path to the medium complexity training ROOT file for curriculum learning.  
  Default: `None`

- `--train_root_file_high`  
  Path to the high complexity training ROOT file for curriculum learning.  
  Default: `None`

- `--output_model`  
  Path where the trained model will be saved.  
  Default: `checkpoints/track_finder.keras`

- `--lr_low`  
  Learning rate for low complexity data.  
  Default: `0.0003`

- `--lr_med`  
  Learning rate for medium complexity data.  
  Default: `0.0001`

- `--lr_high`  
  Learning rate for high complexity data.  
  Default: `0.00003`

- `--factor`  
  Reduction factor for the ReduceLROnPlateau scheduler.  
  Default: `0.3`

- `--patience`  
  Patience for EarlyStopping.  
  Default: `12`

- `--lr_patience`  
  Patience for the learning rate scheduler.  
  Default: `4`

- `--batch_norm`  
  Enable batch normalization (`0` = False, `1` = True).  
  Default: `0`

- `--use_attn`  
  Enable attention mechanism (`0` = False, `1` = True).  
  Default: `0`

- `--use_attn_ffn`  
  Enable feed-forward layers inside attention (`0` = False, `1` = True).  
  Default: `1`

- `--dropout_bn`  
  Dropout rate for the bottleneck layer.  
  Default: `0.0`

- `--dropout_enc`  
  Dropout rate for encoder blocks.  
  Default: `0.0`

- `--dropout_attn`  
  Dropout rate for the attention block.  
  Default: `0.0`

- `--denoise_base`  
  Base number of channels for the denoising U-Net++.  
  Default: `64`

- `--base`  
  Base number of channels for the main U-Net++.  
  Default: `64`

- `--epochs`  
  Number of training epochs.  
  Default: `40`

- `--batch_size`  
  Mini-batch size.  
  Default: `32`

- `--weight_decay`  
  Weight decay coefficient for the AdamW optimizer.  
  Default: `1e-4`

- `--clipnorm`  
  Gradient clipping norm for AdamW.  
  Default: `1.0`

- `--pos_weight`  
  Positive class weight for weighted binary cross entropy.  
  Default: `1.0`

- `--low_ratio`  
  Fraction of total epochs trained on low complexity data.  
  Default: `0.5`

- `--med_ratio`  
  Fraction of total epochs trained on medium complexity data.  
  Default: `0.8`

---

## Loss Function

All models use a **custom loss** combining:

- Sparse categorical cross-entropy (for each muon)
- A penalty on the predicted overlap between mu+ and mu− tracks

Additionally, a weighted BCE loss is used to train the denoiser component.

This encourages the network to separate the two tracks cleanly while matching ground truth.

---

## Output

After training, a Keras `.keras` model file is saved to the `checkpoints/` directory. You can later use this model for inference on test RUS files.

---

## Requirements

- Python 3.9+
- ROOT (PyROOT interface)
- TensorFlow ≥ 2.9
- NumPy, scikit-learn

---

## New Prototypes

`TrackFinder.py` is a prototype script that attempts to jointly train the Track Finder and Momentum models in an end-to-end fashion.

### Details:

- **Architecture**: Combines the U-Net++ based Track Finder with a Momentum model for improved track reconstruction.
- **Training Strategy**: Utilizes a multi-task loss function that balances the objectives of both models.
- **Data Handling**: Efficiently loads and preprocesses data for both tasks, ensuring compatibility and minimizing memory usage.
- **Evaluation**: Implements joint evaluation metrics to assess the performance of both models during training.
- **Usage**: Similar command-line interface as other Track Finder scripts, with additional options for joint training parameters.

### Relevant Architecture Files:

- `backbones.py`: Contains the U-Net++ architecture used in the joint model.
- `data_loaders.py`: Custom data loaders for handling joint training datasets.
- `losses.py`: Defines the multi-task loss function for joint training.
- `layers.py`: Custom layers used in the U-Net++ architecture and axial attention mechanisms.

### Training Command:

```bash
CODEDIR=<path_to_your_code_directory> sbatch scripts/train.slurm
```

- This command submits a job to train the joint model using SLURM workload manager.
- Training scripts are run within apptainer containers for consistent environments.
