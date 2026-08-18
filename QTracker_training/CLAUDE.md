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

# Train TrackFinder locally (single complexity level, no confidence head)
python3 models/TrackFinder.py mc_events_train.root mc_events_val.root \
    --output_model checkpoints/track_finder.keras \
    --batch_norm 1 --use_attn 1 --denoise_base 32 --base 64

# Train TrackFinder with Proposal A confidence head (event-level stop-or-go)
python3 models/TrackFinder.py mc_events_train.root mc_events_val.root \
    --output_model checkpoints/track_finder_conf_a.keras \
    --batch_norm 1 --use_attn 1 --denoise_base 32 --base 64 \
    --confidence_mode event_level --confidence_weight 0.1 --confidence_pos_weight 2.0

# Train TrackFinder with Proposal B confidence head (track-quality F1 overlap)
python3 models/TrackFinder.py mc_events_train.root mc_events_val.root \
    --output_model checkpoints/track_finder_conf_b.keras \
    --batch_norm 1 --use_attn 1 --denoise_base 32 --base 64 \
    --confidence_mode track_quality --confidence_weight 0.1

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
# Also evaluates confidence head if present (auto-detected)
python3 evaluate.py mc_events_val.root checkpoints/track_finder.keras

# Run multi-track finder (auto-regressive, uses confidence-based stopping)
# Requires a model with confidence head for learned stopping; falls back to
# fixed max_steps if loaded model has no confidence head.
cd ../QTracker_main
python3 -c "
from src.models.multi_track_finder import MultiTrackFinder
mtf = MultiTrackFinder(max_steps=10, mode='evaluation', confidence_threshold=0.5)
results = mtf.evaluate('path/to/test_file.root')
print(results)
"

# Evaluate Multi-Track Finder (single-track data works; auto-detected)
python3 eval_multi_track.py mc_events_val.root checkpoints/multi_track_finder.keras

# On HPC (SLURM) — submits eval_multi_track.py via Apptainer
# Results saved to results/multi_track_results.txt
sbatch scripts/eval_multi_track.slurm

# Override CODEDIR or MODEL at submission time:
# CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training \
# MODEL=/mnt/code/checkpoints/multi_track_finder.keras \
# sbatch scripts/eval_multi_track.slurm

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
- **Confidence Head** (optional, controlled by `--confidence_mode`):
  - Architecture: GlobalAveragePooling2D on segmentation backbone features → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
  - `--confidence_mode none` (default): No confidence head, legacy 2-output model
  - `--confidence_mode event_level` (Proposal A): Binary stop-or-go head. Target is 1 if any valid tracks exist in the event, 0 otherwise. Trained with standard BCE loss. Labels are derived automatically from GT hit arrays.
  - `--confidence_mode track_quality` (Proposal B): Track correctness head. Target is the F1 overlap between the model's own segmentation prediction and the best-matching GT track. Computed dynamically inside a custom `train_step` (uses `TrackFinderWithConfidence` model subclass). Trained with BCE against the soft F1 target.
  - `--confidence_weight` controls the loss weighting (default 0.1, start conservatively 0.05–0.2)
  - `--confidence_pos_weight` (Proposal A only) up-weights positive class in BCE for improved recall

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

**models/losses.py**: Custom loss functions (`custom_loss`, `weighted_bce`) for TrackFinder training. Also includes:
  - `compute_track_f1()`: Computes per-event F1 overlap between predicted and GT tracks (used as Proposal B target)
  - `confidence_bce()`: Weighted BCE loss for Proposal A event-level confidence head
  - `confidence_f1_loss()`: Computes F1 target dynamically and applies BCE loss for Proposal B (called inside custom `train_step`)

**models/data_loader.py**: Utilities for loading ROOT files and building hit matrices

**QTracker.py**: Main reconstruction script that orchestrates the full single-track pipeline:

1. Loads detector hits from ROOT file
2. Optionally declusterizes noisy hits (numba-accelerated)
3. Predicts hit arrays using TrackFinder
4. Refines predictions by matching to real hits
5. Predicts momentum for each track
6. Optionally predicts χ² quality metric
7. Writes results to compressed ROOT file

**MultiTrackFinder** (`models/MultiTrackFinder.py`) — *current active approach*:

Single forward pass predicts all dimuon pairs simultaneously. A single model with a shared backbone and `max_pairs` independent output heads — **not** a model ensemble.

---

### MultiTrackFinder: Data and Model Walkthrough

#### What goes in

Each event in the SpinQuest spectrometer produces a set of detector hits. The raw detector has:
- **62 detector layers** (drift chambers, hodoscopes, proportional tubes across stations 1–4)
- **Up to 201 element IDs per layer** (wire/channel positions, i.e. where along the detector a hit landed)

The data loader (`models/data_loader.py`) reads a ROOT file and converts each event into a **2D binary hit matrix** of shape `(62, 201)` — a "picture" of the event where a `1` means that element ID was fired in that detector layer, and `0` means no hit. This becomes the model input after adding a channel dimension: **`(62, 201, 1)`**.

Two versions of this matrix exist per event:
- `X` — noisy: contains both signal muon hits **and** background tracks (0–50 random single-muon tracks per event depending on training phase)
- `X_clean` — clean: contains only the signal muon hits (no background); used as the denoiser supervision target

The ground truth **track labels** (`HitArray_mup`, `HitArray_mum`) record, for each muon in each dimuon pair, which element ID it fired in each of the 62 detector layers. For `max_pairs=3` this is stored as a flat array of length `3 × 62 = 186`, which is reshaped to `(3, 62)`. A value of `0` means the muon did not fire that detector layer (no hit).

After stacking μ⁺ and μ⁻: `y_train` has shape **`(N, max_pairs, 2, 62)`** — one element ID per (event, pair slot, muon charge, detector layer).

#### What the model does

The model is a **two-stage pipeline** trained end-to-end:

**Stage 1 — Denoiser (U-Net++, `denoise_base=32` channels):**

Takes the noisy hit matrix `(batch, 62, 201, 1)` and learns to output a clean version by suppressing background hits. Trained with weighted binary cross-entropy (`pos_weight=20`) against `X_clean`. The `pos_weight` heavily penalizes missed signal hits (false negatives), since failing to keep a real muon hit is worse than keeping a background hit.

U-Net++ uses an encoder-decoder structure with dense skip connections — the encoder progressively downsamples the 2D hit image to extract spatial features at multiple scales, and the decoder reconstructs the full-resolution clean map using those multi-scale features. The `++` dense connections let intermediate decoder nodes aggregate information from all previous encoder/decoder nodes at the same scale, giving better feature reuse than standard U-Net.

**Stage 2 — Segmenter (U-Net++, `base=64` channels, optional AxialAttention):**

Takes the denoised map from Stage 1 and predicts which element ID each muon in each dimuon pair fired per detector layer. Optionally applies AxialAttention (row-wise then column-wise self-attention) after the decoder to capture long-range correlations across the detector.

The segmentation head outputs `max_pairs × 2` channels via a `Conv2D(max_pairs*2, kernel_size=1)`, then reshapes to `(batch, max_pairs, 2, 62, 201)`, and applies softmax over the last axis (201 element IDs). This means for each (pair slot, muon charge, detector layer) triplet, the model produces a probability distribution over all 201 possible element IDs.

**To get a prediction**: take `argmax` over the 201-element softmax → the predicted element ID for each (pair, muon, detector). Residual = predicted − true element ID; exact accuracy = fraction where residual == 0.

#### Loss function

Two loss terms combined:

- `denoise` loss (weighted BCE, weight `3.0` in total): trains Stage 1 to reconstruct the clean hit map
- `segment` loss (weight `1.0`): two components
  1. Masked sparse cross-entropy on element IDs — only computed where ground truth is non-zero (i.e. where the muon actually fired); ignores empty pair slots so the model isn't punished for empty-slot predictions
  2. Presence term (weighted BCE, `lambda_presence=0.2`, `pos_weight_presence=5.0`): trains the model to detect whether a muon hit exists at each position; penalizes false negatives more heavily

#### Output shapes

| Output | Shape | Meaning |
|--------|-------|---------|
| `denoise` | `(batch, 62, 201, 1)` | Predicted clean hit matrix |
| `segment` (softmax) | `(batch, max_pairs, 2, 62, 201)` | Per-element-ID probability for each pair/charge/detector |
| `segment` (argmax) | `(batch, max_pairs, 2, 62)` | Predicted element ID — the actual track prediction |

#### Pair slot ordering

Slots 0, 1, 2 are filled sequentially by the data generator (`gen_training_random.py`). Pair 0 always has the most events (44k valid), Pair 1 fewer (36k), Pair 2 fewest (27k), reflecting how many events truly have 2 or 3 dimuon pairs. The Hungarian matching evaluation (`find_best_permutation` in `eval_multi_track.py`) handles any slot-ordering ambiguity by finding the optimal assignment of predicted slots to ground-truth pairs per event.

#### Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| Exact accuracy | Fraction of detectors where predicted elementID == true elementID |
| Within-2 accuracy | Fraction where \|predicted − true\| ≤ 2 channels |
| Mean residual | Average \|predicted − true\| in channel units |
| Existence F1 | Whether the model correctly detects that a pair slot is occupied |
| Chi-squared | Normalized residual sum — large values indicate poor fit |

Current results (max_pairs=3, Hungarian matching):
- Exact acc: 42.2% / 42.7% (μ⁺/μ⁻)
- Within-2: 51.7% / 54.1%
- Mean residual: 14.8 / 6.8 channels
- μ⁺ Station 2 (det 19–30) residuals elevated (~50 ch mean) — active investigation area

---

**Architecture:**
- Input: binary hit matrix `(batch, 62, 201, 1)`
- Stage 1 — Denoiser (U-Net++, `denoise_base=32`): removes background tracks, outputs clean hit matrix
- Stage 2 — Segmenter (U-Net++ with optional AxialAttention, `base=64`): takes denoised output, produces `max_pairs` pair predictions
- Output 1 `denoise`: `(batch, 62, 201, 1)` — cleaned hit matrix
- Output 2 `segment`: `(batch, max_pairs, 2, 62, 201)` — softmax over 201 elementIDs per (pair, muon charge, detector layer)
- ~48M parameters total

**Loss:**
- `denoise`: weighted BCE (`pos_weight=20.0`) — heavily penalizes missing true hits
- `segment`: multi-track loss = masked sparse cross-entropy over elementIDs + presence detection term (`lambda_presence=0.2`, `pos_weight_presence=5.0`) for empty pair slots
- Loss weights: `denoise × 3.0 + segment × 1.0`

**Training configuration (current):**
- 1× A100 GPU, `batch_size=32`, mixed precision (FP16)
- Curriculum learning: low (0–16 bg tracks, epochs 1–30) → med (17–33, epochs 31–48) → high (34–50, epochs 49–60)
- `lr_low=3e-4`, `lr_med=1e-4`, `lr_high=3e-5`
- `ReduceLROnPlateau(factor=0.5, patience=5)`, `EarlyStopping(patience=20)` on high phase only
- Container: `TfRootBuild_v2.sif` (v1 has CUDA 13.0 double-registration bug — do NOT use)

**Metrics (per epoch):**
- `denoise_precision`, `denoise_recall` — hit-level denoising quality
- `segment_nonempty_acc` — accuracy on non-empty pair slots only (true elementID ≠ 0); honest track-finding metric
- `segment_mean_residual` — mean |predicted_elementID − true_elementID| on non-empty slots (in detector elements); target <5 for good reconstruction
- DO NOT use `segment_accuracy` — inflated by empty pair slots

**Expected outputs after full training:**
- `checkpoints/multi_track_finder_best.keras` — best val_loss checkpoint (saved per epoch)
- `checkpoints/multi_track_finder.keras` — final model after all curriculum phases
- MLflow run in experiment `multi_track_v2` at `/project/ptgroup/spinquest/Anvesh/mlruns/`

**Data:**
- `/project/ptgroup/spinquest/Anvesh/data/multi_track/processed_files/`
  - `mc_events_train_low/med/high.root` — training at 3 complexity levels
  - `mc_events_val.root` — validation (full complexity range)

**Known issues resolved:**
- `K.set_value()` → use `model.optimizer.learning_rate.assign()` for Keras3
- MLflow `active_run()` thread-local bug with MirroredStrategy → use `MlflowClient` with explicit `run_id`
- `jit_compile=True` breaks `model.save()` with custom loss closures — removed
- `reshape(max_pairs, 62)` on data generated with old `max_pairs=5` — fix: slice `[:max_pairs*62]` before reshape in `data_loader.py`

**Auto-regressive Multi-Track Finder** (`QTracker_main/src/models/multi_track_finder.py`) — *alternative approach, not in use*:
- Iteratively invokes single-track TrackFinder with hit subtraction between steps
- Requires confidence head for learned stopping; falls back to fixed `max_steps`

**refine.py**: Matches predicted element IDs to actual recorded hits by finding closest matches

### Important Settings in QTracker.py

```python
USE_CHI2 = False          # Must be False for first run (before training χ² model)
USE_DECLUSTERING = False  # Enable to clean clustered/noisy hits
USE_SMAXMATRIX = False    # Enable to save softmax probability matrices
```

### Important Settings for Multi-Track Finder

```python
# In QTracker_main/src/config.py
MAX_STEPS = 10             # Maximum auto-regressive iterations (should match data generation)

# When constructing MultiTrackFinder:
MultiTrackFinder(
    max_steps=10,                  # Should match MAX_STEPS / data generation
    mode="evaluation",             # "evaluation" or "production"
    confidence_threshold=0.5,      # Confidence below this marks event inactive
    model_path="checkpoints/track_finder.keras",  # Optional, defaults to config
)
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

1. **Branch naming**: Current work is on `dev` branch. Main branch is `main`.

2. **Detector coordinate system**:
   - 62 detector layers (stations 1-4 with drift chambers, hodoscopes, proportional tubes)
   - 201 element IDs per detector (wire/channel positions)
   - Unused stations: 7-12, 55-58, 59-62 (masked in momentum prediction)

3. **Data preprocessing order**: Always follow the sequence in `scripts/preprocess.slurm`:
   - Split → Skim → Separate → Combine → Generate → Inject backgrounds → Inject noise

4. **Model training order**:
   - **Joint multi-track path** (current): Train `MultiTrackFinder.py` directly — no prior single-track model needed
   - **Auto-regressive path** (alternative): Train `TrackFinder.py` with `--confidence_mode event_level` or `track_quality` first, then use via `QTracker_main`
   - Momentum models trained on clean truth data (no interdependence with either track finder)
   - χ² model requires output from both TrackFinder and Momentum models

5. **Evaluation workflow**:
   - TrackFinder: Residual distributions (difference between predicted and true element IDs)
   - Confidence head (auto-detected by `evaluate.py`): Binary accuracy/precision/recall, Pearson correlation between confidence score and F1 overlap, scatter plots of confidence vs reconstruction quality
   - Momentum: Momentum residuals, invariant mass spectrum (should show J/ψ peak at ~3.1 GeV)
   - Quality: χ² distribution, correlation with reconstruction accuracy
   - Multi-track: Per-step residuals, track duplication rate, early stopping behavior

6. **File size considerations**: Training datasets can be very large (500K-19M events). Use `data/skim.py` or `data/skim_flat.py` to create manageable subsets for development.

7. **Mixed precision**: Enabled by default in TrackFinder for performance. Final output layers cast to FP32 for numerical stability.
