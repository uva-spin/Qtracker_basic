# Single RUS Track and Momentum Reconstruction with QTracker_training

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
conda install -c conda-forge numpy tensorflow numba
```

Ensure that your ROOT installation includes PyROOT support.

---

## Usage

Run the reconstruction script with:

```bash
python3 QTracker.py path/to/input.root --output_file path/to/output.root
```

Arguments:

- `root_file` (required): Path to the input ROOT file containing detector hit information.
- `--output_file` (optional): Path to the output ROOT file. Defaults to `qtracker_reco.root`.

---

## Model Files

This script expects pre-trained TensorFlow `.h5` model files stored in the `./checkpoints/` directory:

| Purpose                 | Model Path                            | Description                                           |
| :---------------------- | :------------------------------------ | :---------------------------------------------------- |
| Track Finder            | `checkpoints/track_finder.keras`      | Predicts the hit patterns of muon tracks.             |
| Momentum Predictor (μ⁺) | `checkpoints/mom_mup.h5`              | Predicts momentum `(pₓ, pᵧ, p_z)` for positive muons. |
| Momentum Predictor (μ⁻) | `checkpoints/mom_mum.h5`              | Predicts momentum `(pₓ, pᵧ, p_z)` for negative muons. |
| χ² Metric Model         | `checkpoints/chi2_predictor_model.h5` | (Optional) Predicts a track quality χ² metric.        |

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

Modify the following settings at the top of `QTracker.py` to control behavior:

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

- The TrackFinder model requires a custom axial attention layer (`AxialAttention`) for loading.
- Track momentum predictions mask out unused detector stations automatically.
- If you wish to generate new training data for the χ² model, you must first run the script with `USE_CHI2 = False`.

---

## Example

```bash
python3 QTracker.py input_data.root --output_file reco_output.root
```

---

## TrackFinder Training and Evaluation

First, preprocess the target ROOT file by injecting noise to generate messy hit matrices. To achieve this, run:

```bash
sbatch scripts/preprocess.slurm
```

To train and evaluate TrackFinder models, run the following script:

```bash
CODEDIR=/path/to/your/code/ sbatch scripts/train.slurm
```

To train and evaluate Momentum models, run the following script:

```bash
sbatch scripts/momentum.slurm
```

This script is designed to preprocess input ROOT files, train currently available TrackFinder models, and evaluate each of them using a distribution of residuals. The evaluation logic can be found under `evaluate.py`. Directly modify the shell script to train your custom TrackFinder model or use it as reference.

---

## Multi-Track Reconstruction

Two approaches exist for reconstructing multiple dimuon pairs per event:

### Approach 1: Ensemble MultiTrackFinder (current, `models/MultiTrackFinder.py`)

Predicts all dimuon pairs simultaneously in a single forward pass using an ensemble architecture: a shared U-Net++ denoising backbone feeding into `max_pairs` independent segmentation heads, one per pair slot.

| | Single-Track (`TrackFinder.py`) | Ensemble Multi-Track (`MultiTrackFinder.py`) |
|---|---|---|
| Pairs per forward pass | 1 | Up to `max_pairs` (default 5) |
| Architecture | 2 U-Net++ backbones | Shared denoiser + independent per-pair heads |
| Confidence head | Optional | None |
| Stopping mechanism | Confidence threshold | N/A — all pairs in one shot |
| Hit subtraction | Soft subtraction between steps | None |
| Model size | ~24M params | ~48M params |
| Output shape | `(batch, 2, 62, 201)` | `(batch, max_pairs, 2, 62, 201)` |
| Loss | Weighted BCE + overlap penalty | Weighted BCE + presence detection per pair |

**Why independent heads?** A single joint decoder struggles to attribute hits across multiple indistinguishable pair slots — the loss is ambiguous about which predicted slot corresponds to which true track. Independent heads give each slot its own gradient signal, resolving this ambiguity.

**Architecture details:**
- Input: binary hit matrix `(batch, 62, 201, 1)`
- Stage 1 — Denoiser (U-Net++, `denoise_base=32`): suppresses background tracks, outputs clean hit matrix
- Stage 2 — Segmenter (U-Net++ with AxialAttention, `base=64`): `max_pairs` independent heads, each outputs softmax over 201 elementIDs per detector per muon charge
- Output `denoise`: `(batch, 62, 201, 1)`
- Output `segment`: `(batch, max_pairs, 2, 62, 201)` — softmax probabilities over elementIDs

**Loss functions:**
- Denoiser: weighted BCE (`pos_weight=20`) with loss weight 3.0
- Segmenter: sparse cross-entropy over elementIDs + presence detection term for empty slots, loss weight 1.0

**Training:**
```bash
cd /scratch/am4qw/Qtracker_basic/QTracker_training
git pull origin AnveshTrackFinderWork
sbatch scripts/train_multi.slurm
```

Curriculum learning: low complexity (0–16 background tracks) → medium (17–33) → high (34–50), across 60 total epochs.

**Evaluation:**
```bash
sbatch scripts/eval_multi_track.slurm
```

**Checkpoints:**
- `checkpoints/multi_track_finder_best.keras` — best val_loss model, saved each epoch
- `checkpoints/multi_track_finder.keras` — final model after all curriculum phases

**MLflow tracking** (experiment `multi_track_v2`):
```bash
# View on Rivanna (requires SSH tunnel to the specific login node):
apptainer exec ... python3 -m mlflow ui \
    --backend-store-uri file:///project/ptgroup/spinquest/Anvesh/mlruns \
    --host 127.0.0.1 --port 5000
```

---

### Metrics

> **Do NOT use `segment_accuracy`** — it is inflated by empty pair slots (events with fewer than `max_pairs` true pairs have padded slots with elementID=0, which are trivially predicted correctly).

Use these instead:

| Metric | Description | What to look for |
|--------|-------------|-----------------|
| `segment_nonempty_acc` | Accuracy on non-empty pair slots only (true elementID ≠ 0) | Increasing over training |
| `segment_mean_residual` | Mean \|predicted − true elementID\| on non-empty slots (in detector channels) | Decreasing; <5 is good reconstruction |
| `denoise_recall` | Fraction of true signal hits recovered by denoiser | Should be >0.99 |
| `denoise_precision` | Fraction of denoiser predictions that are true hits | Increasing with training |
| `val_loss` | Combined validation loss across denoiser + segmenter | Primary training signal |

---

### Approach 2: Auto-regressive Multi-Track (`QTracker_main/`)

Runs the single-track `TrackFinder` iteratively, subtracting each found track from the hit matrix before the next step. Requires a confidence head for learned stopping.

See `QTracker_main/README.md` for usage.
