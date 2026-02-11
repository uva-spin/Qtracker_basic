# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QTracker is a deep learning framework for reconstructing and analyzing muon tracks in particle physics experiments (SpinQuest at Fermilab). The system uses a two-stage neural network pipeline:

1. **Track Finder**: U-Net++ with axial attention to identify μ⁺ and μ⁻ tracks from noisy detector hit matrices
2. **Momentum Predictor**: Separate DNNs for predicting 3-momentum (px, py, pz) for each muon
3. **Quality Metric**: χ² predictor for track quality assessment

## Common Commands

### Data Preprocessing

```bash
# Full preprocessing pipeline (SLURM on HPC cluster)
sbatch scripts/preprocess.slurm

# Or run individual steps locally:
python3 data/separate.py JPsi_Target.root
python3 data/gen_training.py JPsi_Target_track1.root JPsi_Target_track2.root
python3 data/combine.py MUP_Dump.root MUM_Dump.root --output single_muons.root
python3 data/messy_gen.py finder_training.root single_muons.root
python3 data/noisy_gen.py mc_events.root
```

### Training Models

```bash
# Train TrackFinder (SLURM)
CODEDIR=/path/to/code sbatch scripts/train.slurm

# Train Momentum models (SLURM)
sbatch scripts/momentum.slurm

# Train TrackFinder locally (single complexity level)
python3 models/TrackFinder.py mc_events_train.root mc_events_val.root \
    --output_model checkpoints/track_finder.keras \
    --batch_norm 1 --use_attn 1 --denoise_base 32 --base 64

# Train Momentum models locally
python3 models/Momentum_training.py momentum_training-1.root --output checkpoints/mom_mup.h5
python3 models/Momentum_training.py momentum_training-2.root --output checkpoints/mom_mum.h5

# Train Quality Metric model
python3 models/Qmetric_training.py qtracker_reco.root
```

### Evaluation and Testing

```bash
# Run reconstruction on validation data
python3 QTracker.py mc_events_val.root --output_file qtracker_reco.root

# Evaluate TrackFinder performance (residual distributions)
python3 evaluate.py mc_events_val.root checkpoints/track_finder.keras

# Evaluate Momentum reconstruction
python3 evaluate_momentum.py qtracker_reco.root

# Plot invariant mass spectrum (verify J/ψ peak)
python3 Util/imass_plot.py qtracker_reco.root --output_plot invariant_mass.png

# Visualize detector hit matrices for specific event
python3 Util/plot_HitMatrix.py yourfile.root -event 42
```

### Container and Deployment

```bash
# Build Apptainer image (requires sudo)
apptainer build TfRootBuild.sif build/TfRootBuild.def

# Deploy to HPC
bash build/deploy.sh
```

### Code Quality

```bash
# Run pre-commit hooks (ruff linting and formatting)
pre-commit run --all-files
```

## Architecture

### Data Flow

```
Raw ROOT Files (RUS format)
  ↓
separate.py → Split dimuons into μ⁺/μ⁻ tracks
  ↓
gen_training.py → Generate hit arrays and training files
  ↓
combine.py → Merge single-muon background files
  ↓
messy_gen.py → Inject background tracks (0-50 tracks/event)
  ↓
noisy_gen.py → Inject electronic/cluster noise
  ↓
Preprocessed Training Data
  ↓
TrackFinder.py → Train denoiser-segmenter pipeline
  ↓
QTracker.py → Run reconstruction (predict hit arrays)
  ↓
refine.py → Match predictions to real detector hits
  ↓
Momentum_training.py → Train momentum predictors
  ↓
Qmetric_training.py → Train χ² quality metric
  ↓
Final Reconstructed Tracks
```

### Model Architecture Details

**TrackFinder (models/TrackFinder.py)**:

- This model identifies dimuon pair tracks from a hit matrix by first denoising background noise tracks, then classifying and matching hits per detector to corresponding mu+ and mu- pairs
- Two-stage U-Net++ pipeline trained end-to-end
- Stage 1 (Denoiser): Removes background hits from noisy events
- Stage 2 (Segmenter): Extracts μ⁺/μ⁻ tracks with axial attention
- Input: Binary hit matrix (62 detectors × 201 elements)
- Output: Softmax probabilities over 201 element IDs per detector, per muon
- Custom loss: Sparse categorical cross-entropy + overlap penalty
- Supports curriculum learning (low/med/high complexity datasets)
- Mixed precision training (FP16) enabled by default

**Momentum Models (models/Momentum_training.py)**:

- Fully connected DNNs with batch normalization and dropout
- Input: Hit arrays (62 detectors, 2 features: elementID + driftDistance)
- Output: 3-momentum vector (px, py, pz)
- Trained separately for μ⁺ and μ⁻
- Masks unused detector stations (7-12, 55-58, 59-62)

**Quality Metric (models/Qmetric_training.py)**:

- Predicts χ² value for track quality assessment
- Input: Hit arrays + momentum vectors
- Uses L2 regularization and dropout

### Key Components

**models/backbones.py**: Contains `unetpp_backbone()` - the core U-Net++ architecture used by TrackFinder

**models/layers.py**: Custom layers including `AxialAttention` (required for loading TrackFinder models)

**models/losses.py**: Custom loss functions (`custom_loss`, `weighted_bce`) for TrackFinder training

**models/data_loader.py**: Utilities for loading ROOT files and building hit matrices

**QTracker.py**: Main reconstruction script that orchestrates the full pipeline:

1. Loads detector hits from ROOT file
2. Optionally declusterizes noisy hits (numba-accelerated)
3. Predicts hit arrays using TrackFinder
4. Refines predictions by matching to real hits
5. Predicts momentum for each track
6. Optionally predicts χ² quality metric
7. Writes results to compressed ROOT file

**refine.py**: Matches predicted element IDs to actual recorded hits by finding closest matches

### Important Settings in QTracker.py

```python
USE_CHI2 = False          # Must be False for first run (before training χ² model)
USE_DECLUSTERING = False  # Enable to clean clustered/noisy hits
USE_SMAXMATRIX = False    # Enable to save softmax probability matrices
```

### Model Checkpoints Location

All trained models are stored in `checkpoints/`:

- `track_finder.keras` - TrackFinder model (requires AxialAttention custom object)
- `mom_mup.h5` - Momentum predictor for μ⁺
- `mom_mum.h5` - Momentum predictor for μ⁻
- `chi2_predictor.h5` - Quality metric predictor (optional)

### ROOT File Format (RUS)

All ROOT files contain a TTree named `"tree"` with vector branches:

- Hit-level: `detectorID`, `elementID`, `driftDistance`, `tdcTime`
- Track-level: `trackID`, `gCharge`, `gpx/gpy/gpz` (truth momentum)
- Muon ID: `muID` (1 for μ⁺, 2 for μ⁻)
- Ground truth arrays: `HitArray_mup`, `HitArray_mum` (shape: 62)
- Reconstructed arrays: `qHitArray_mup`, `qHitArray_mum` (output from QTracker.py)

### Curriculum Learning Strategy

TrackFinder supports training on three complexity levels sequentially:

- **Low**: 0-16 background tracks per event (first 50% of epochs)
- **Medium**: 17-33 background tracks per event (next 30% of epochs)
- **High**: 34-50 background tracks per event (final 20% of epochs)

Specify multiple training files with `--train_root_file_med` and `--train_root_file_high` flags.

## Development Environment

### Dependencies

- Python 3.9+
- ROOT 6.32+ (with PyROOT)
- TensorFlow 2.19+ (GPU support recommended)
- NumPy, scikit-learn, matplotlib, uproot, numba

Install via conda:

```bash
conda install -c conda-forge numpy tensorflow uproot sklearn ROOT numba
```

### HPC/SLURM Usage

All SLURM scripts expect:

- `CODEDIR` environment variable set to repository path
- Apptainer container image at specified `IMAGE` path
- Code mounted to `/mnt/code` inside container

GPU resources requested:

- TrackFinder training: 4× A100 GPUs, 256GB RAM, 72hrs
- Momentum training: 1× GPU, 256GB RAM, 72hrs

### Linting and Formatting

Pre-commit hook configured with ruff (v0.14.4):

- Auto-fixes linting issues
- Auto-formats code
- Run manually: `pre-commit run --all-files`

## Project-Specific Conventions

1. **Branch naming**: Current work is on `feature/mixed-precision` branch. Main branch is `main`.

2. **Detector coordinate system**:
   - 62 detector layers (stations 1-4 with drift chambers, hodoscopes, proportional tubes)
   - 201 element IDs per detector (wire/channel positions)
   - Unused stations: 7-12, 55-58, 59-62 (masked in momentum prediction)

3. **Data preprocessing order**: Always follow the sequence in `scripts/preprocess.slurm`:
   - Split → Skim → Separate → Combine → Generate → Inject backgrounds → Inject noise

4. **Model training order**:
   - TrackFinder must be trained first
   - Momentum models trained on clean truth data (no interdependence with TrackFinder yet)
   - χ² model requires output from both TrackFinder and Momentum models

5. **Evaluation workflow**:
   - TrackFinder: Residual distributions (difference between predicted and true element IDs)
   - Momentum: Momentum residuals, invariant mass spectrum (should show J/ψ peak at ~3.1 GeV)
   - Quality: χ² distribution, correlation with reconstruction accuracy

6. **File size considerations**: Training datasets can be very large (500K-19M events). Use `data/skim.py` or `data/skim_flat.py` to create manageable subsets for development.

7. **Mixed precision**: Enabled by default in TrackFinder for performance. Final output layers cast to FP32 for numerical stability.
