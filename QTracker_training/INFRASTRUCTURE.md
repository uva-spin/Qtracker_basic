# QTracker Infrastructure & MLOps Guide

## Table of Contents
1. [Infrastructure Overview](#infrastructure-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation Pipeline](#evaluation-pipeline)
6. [Visualization Tools](#visualization-tools)
7. [MLOps Workflow](#mlops-workflow)

---

## Infrastructure Overview

### Computing Resources

#### **Rivanna HPC (University of Virginia)**
- **Account**: `am4qw` (spinquest group)
- **Storage Locations**:
  - **Scratch space**: `/scratch/am4qw/Qtracker_basic/` (working directory for development)
  - **Project storage**: `/project/ptgroup/spinquest/` (shared data/models)
  - **Anvesh's area**: `/project/ptgroup/spinquest/Anvesh/`
  - **Donghwa's area**: `/project/ptgroup/spinquest/Donghwa/` (trained models & data)

#### **Container Infrastructure**
- **Technology**: Apptainer (formerly Singularity)
- **Image Location**: `/project/ptgroup/spinquest/David/TfRootBuild.sif`
- **Purpose**: Provides consistent environment with TensorFlow 2.x, ROOT, and all dependencies
- **Why containers?**: Ensures reproducibility across different compute nodes and users

#### **Job Scheduling**
- **System**: SLURM (Simple Linux Utility for Resource Management)
- **Partition**: `standard` or `gpu` (for GPU training)
- **Account**: `spinquest`
- **Typical resources**:
  - Training: 4-8 GPUs, 32-64GB RAM, 12-48 hours
  - Evaluation: 1 GPU, 8-16GB RAM, 1-4 hours
  - Visualization: CPU only, 8GB RAM, 30 minutes

---

## Data Pipeline

### Data Format: ROOT Files

**What is ROOT?**
- High-energy physics data analysis framework (CERN)
- Binary file format (`.root`) containing structured event data
- Each "event" = one particle collision/decay in the detector

### Directory Structure

```
data/
├── raw_files/           # Original detector data from experiments
├── processed_files/     # Preprocessed data ready for training
│   ├── mc_events_train.root  # Training set (Monte Carlo simulation)
│   ├── mc_events_val.root    # Validation set
│   ├── mc_events_test.root   # Test set
│   └── finder_training_*.root # Track finder training data
└── multi_track/         # Multi-track event datasets
    └── mc_multitrack_*.root
```

### Data Generation Scripts

#### **`data/gen_training.py`**
**Purpose**: Generate training data from raw ROOT files
**What it does**:
- Reads detector hit information from ROOT files
- Creates hit matrices (62 detectors × 201 elements per detector)
- Generates true track labels for supervised learning
- Adds realistic noise and background hits
- Splits into train/val/test sets

**Key parameters**:
```python
n_events = 100000        # Number of events to generate
noise_level = 0.05       # Background hit probability
detector_efficiency = 0.95  # Detector hit detection rate
```

#### **`data/messy_gen.py`**
**Purpose**: Generate "messy" multi-track events
**What it does**:
- Creates events with 2-5 overlapping tracks
- Simulates challenging scenarios (close tracks, shared hits)
- Tests model's ability to separate overlapping tracks

#### **`data/noisy_gen.py`**
**Purpose**: Add realistic detector noise
**What it does**:
- Thermal noise (random hits)
- Cross-talk between detector channels
- Dead/hot channels
- Pile-up effects (multiple collisions)

#### **`data/separate.py`**
**Purpose**: Split composite ROOT files
**What it does**:
- Separates μ+ and μ- tracks into different files
- Useful for momentum-specific training

#### **`data/skim.py` & `data/skim_flat.py`**
**Purpose**: Filter and reduce dataset size
**What it does**:
- Removes empty events
- Filters by quality criteria (min hits, track quality)
- Flattens nested structures for faster I/O

---

## Model Architecture

### Core Models

#### **`models/TrackFinder.py`** (Main Model)

**Architecture**: Dual U-Net++ with Axial Attention

```
Input: Hit Matrix [62 detectors × 201 elements]
  ↓
┌─────────────────────────────────┐
│  Denoising U-Net++              │
│  - Removes background noise     │
│  - Axial attention layers       │
│  - Output: Cleaned hit matrix   │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Segmentation U-Net++           │
│  - Identifies track pixels      │
│  - Separates μ+ and μ- tracks   │
│  - Output: [μ+, μ-] masks       │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Optional Confidence Head       │
│  - Global Average Pooling       │
│  - Dense layers (128 → 64 → 1) │
│  - Sigmoid activation           │
└─────────────────────────────────┘
  ↓
Output: [μ+ mask, μ- mask, confidence?]
```

**Three Confidence Modes**:

1. **`confidence_mode="none"`** (Original)
   - Two outputs: denoise + segment
   - No learned stopping criterion
   
2. **`confidence_mode="event_level"`** (Proposal A)
   - Binary confidence: "Are there more tracks to find?"
   - Output: [denoise, segment, stop_probability]
   - Training target: 1 if tracks remain, 0 if empty
   
3. **`confidence_mode="track_quality"`** (Proposal B)
   - F1-score prediction: "How good is this track?"
   - Output: [denoise, segment, expected_f1]
   - Training target: F1 overlap between prediction and truth
   - Allows quality-based thresholding

#### **`models/MultiTrackFinder.py`**

**Purpose**: Auto-regressive multi-track finding

**How it works**:
```python
# Pseudo-code
for iteration in range(max_steps):
    # 1. Find best track in current hit matrix
    pred_mup, pred_mum, confidence = model.predict(hit_matrix)
    
    # 2. Check stopping criterion
    if confidence < threshold:
        break  # No more tracks to find
    
    # 3. Soft track subtraction
    hit_matrix = hit_matrix - soft_subtract(pred_mup, pred_mum)
    
    # 4. Store this track
    tracks.append((pred_mup, pred_mum))
```

**Key innovation**: Soft subtraction removes found tracks while preserving remaining track information

#### **`models/layers.py`**

**Custom Layers**:

- **`AxialAttention`**: Efficient attention mechanism
  - Separates row-wise and column-wise attention
  - Reduces complexity from O(N²) to O(N√N)
  - Captures long-range detector correlations

- **`ConvBlock`**: Standard convolution + batch norm + activation

- **`UNetPPEncoder/Decoder`**: Dense skip connections
  - Multiple pathways between encoder/decoder
  - Better gradient flow than standard U-Net

#### **`models/backbones.py`**

**Purpose**: Alternative encoder architectures
**Options**:
- ResNet50 backbone (pre-trained on ImageNet)
- EfficientNet backbone
- Custom lightweight backbone for faster inference

---

## Training Pipeline

### Training Scripts

#### **`models/Momentum_training.py`**

**Purpose**: Train momentum reconstruction network
**What it does**:
- Takes track masks → predicts particle momentum
- Separate networks for μ+ and μ-
- Uses track curvature in magnetic field
- Loss: Mean Squared Error on momentum

#### **`models/Qmetric_training.py`**

**Purpose**: Train χ² quality predictor
**What it does**:
- Predicts track fit quality (χ²/ndf)
- Used to filter bad tracks
- Helps reject fake tracks in multi-track scenarios

#### **Main Training Script** (Inferred from SLURM scripts)

**Curriculum Learning Strategy**:
```python
# Phase 1: Low complexity (1-2 tracks, low noise)
train_phase_1 = {
    'n_tracks': (1, 2),
    'noise_level': 0.01,
    'epochs': 50
}

# Phase 2: Medium complexity (2-3 tracks, medium noise)
train_phase_2 = {
    'n_tracks': (2, 3),
    'noise_level': 0.05,
    'epochs': 50
}

# Phase 3: High complexity (3-5 tracks, high noise)
train_phase_3 = {
    'n_tracks': (3, 5),
    'noise_level': 0.10,
    'epochs': 100
}
```

**Training Parameters** (from `config.py`):
```python
BATCH_SIZE = 16          # Per GPU
LEARNING_RATE = 1e-4     # Adam optimizer
WEIGHT_DECAY = 1e-5      # L2 regularization
EPOCHS = 200             # Total epochs
EARLY_STOPPING = 20      # Patience for early stopping
```

**Multi-GPU Training**:
```python
# Uses TensorFlow's MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)
```

### Loss Functions (`models/losses.py`)

#### **Segmentation Loss**
```python
loss = dice_loss + binary_crossentropy
```
- Dice loss: Handles class imbalance (few track pixels vs. many background)
- BCE: Provides stable gradients

#### **Confidence Loss (Proposal A)**
```python
confidence_loss = binary_crossentropy(y_true_has_tracks, y_pred_confidence)
```

#### **Confidence Loss (Proposal B)**
```python
# Target: F1 score of predicted track vs. ground truth
f1_target = 2 * (precision * recall) / (precision + recall)
confidence_loss = mse(f1_target, y_pred_confidence)
```

---

## Evaluation Pipeline

### **`evaluate.py`**

**Purpose**: Comprehensive model evaluation

**What it does**:
1. **Single-track metrics**:
   - Precision, Recall, F1 per track
   - Hit efficiency (% of true hits recovered)
   - Purity (% of predicted hits that are true)

2. **Multi-track metrics**:
   - Track finding efficiency (% events with all tracks found)
   - Ghost rate (fake tracks per event)
   - Track assignment accuracy

3. **Confidence head evaluation** (if present):
   - Binary classification metrics (Proposal A)
   - F1 correlation (Proposal B)
   - Optimal threshold analysis

**Usage**:
```bash
sbatch scripts/evaluate.slurm  # Run as SLURM job
```

### **`eval_multi_track.py`**

**Purpose**: Multi-track specific evaluation

**What it does**:
- Tests auto-regressive track finding
- Measures performance vs. number of tracks
- Analyzes stopping criterion effectiveness

### **`evaluate_momentum.py`**

**Purpose**: Momentum reconstruction evaluation

**Metrics**:
- Momentum resolution: σ(p_rec - p_true) / p_true
- Momentum bias: mean(p_rec - p_true)
- Charge identification accuracy

---

## Visualization Tools

### **`Util/visualize_single_track.py`**

**Purpose**: Create 5-frame animation of single-track pipeline

**Frames**:
1. **Raw Input**: Original detector hits (truth + noise)
2. **Denoised**: After denoising U-Net
3. **Segmentation**: Predicted track masks (μ+, μ-)
4. **Overlay**: Prediction overlaid on truth
5. **Results**: Metrics and statistics

**Usage**:
```bash
python3 Util/visualize_single_track.py \
    data.root model.keras \
    --event 0 --format mp4
```

### **`Util/visualize_multitrack.py`**

**Purpose**: Animate auto-regressive multi-track finding

**Shows**:
- Step-by-step track finding
- Hit matrix evolution (soft subtraction)
- Confidence scores per iteration
- Cumulative track predictions

**Output**: 4-panel animation
- Panel 1: Current hit matrix + predictions
- Panel 2: μ+ softmax confidence map
- Panel 3: μ- softmax confidence map
- Panel 4: Metrics (F1, precision, recall, confidence)

---

## MLOps Workflow

### Development Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT (Mac)                   │
│  - Write code in VSCode                                     │
│  - Test small changes                                       │
│  - Git commit + push to GitHub                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              RIVANNA HPC (Development Branch)                │
│  - Git pull latest changes                                  │
│  - Submit SLURM jobs for training                           │
│  - Monitor with: squeue -u am4qw                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING & EVALUATION                     │
│  - Training: scripts/preprocess.slurm → train.slurm         │
│  - Checkpoints saved to: checkpoints/                       │
│  - TensorBoard logs: logs/                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   MODEL VALIDATION                           │
│  - Run evaluate.slurm                                       │
│  - Generate visualizations                                  │
│  - Review results in plots/                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT (if good)                       │
│  - Copy best model to /project/ptgroup/spinquest/Anvesh/    │
│  - Update model registry                                    │
│  - Merge dev branch → main                                  │
└─────────────────────────────────────────────────────────────┘
```

### File Organization Best Practices

**On Rivanna**:
```
/scratch/am4qw/Qtracker_basic/        # Your working directory
├── QTracker_training/                 # Development code (git repo)
│   ├── models/                        # Model definitions
│   ├── data/                          # Preprocessing scripts
│   ├── scripts/                       # SLURM job scripts
│   └── Util/                          # Utility scripts
└── checkpoints/                       # Training checkpoints (local)

/project/ptgroup/spinquest/Anvesh/     # Your persistent storage
├── models/                            # Production-ready models
│   ├── track_finder_v1.keras
│   └── track_finder_v2_with_conf.keras
├── data/                              # Curated datasets
└── results/                           # Evaluation results

/project/ptgroup/spinquest/Donghwa/    # Reference models/data
└── Qtracker_basic/QTracker_training/
    ├── checkpoints/
    │   ├── track_finder_flagship.keras  # Best single-track model
    │   └── track_finder.keras           # Standard model
    └── data/processed_files/
        ├── mc_events_val.root
        └── mc_events_test.root
```

### SLURM Script Templates

**Location**: `QTracker_training/scripts/`

- **`preprocess.slurm`**: Data preprocessing (ROOT → TFRecord)
- **`train.slurm`**: Model training with multi-GPU
- **`evaluate.slurm`**: Model evaluation
- **`visualize.slurm`**: Generate animations
- **`submit_viz.sh`**: Helper script for easy job submission

**Common SLURM parameters**:
```bash
#SBATCH --account=spinquest      # Your allocation
#SBATCH --partition=gpu          # GPU partition for training
#SBATCH --gres=gpu:v100:4        # 4 V100 GPUs
#SBATCH --mem=64GB               # Memory
#SBATCH --time=12:00:00          # 12 hours
#SBATCH --output=logs/%j.out     # Output log
#SBATCH --error=logs/%j.err      # Error log
```

### Git Workflow

**Branches**:
- `main`: Stable production code
- `AnveshTrackFinderWork`: Your development branch

**Typical workflow**:
```bash
# On Mac: Make changes
git add <files>
git commit -m "Description"
git push origin AnveshTrackFinderWork

# On Rivanna: Pull and test
cd /scratch/am4qw/Qtracker_basic/QTracker_training
git pull origin AnveshTrackFinderWork
sbatch scripts/train.slurm

# After validation: Merge to main
git checkout main
git merge AnveshTrackFinderWork
git push origin main
```

---

## Key Experiments & Ablations

### Experiment 1: Confidence Head Comparison
**Question**: Which confidence mechanism works better?

**Setup**:
- Train 3 models: none, event_level, track_quality
- Evaluate on multi-track test set
- Compare: stopping accuracy, track quality, inference speed

### Experiment 2: Curriculum Learning
**Question**: Does curriculum learning improve multi-track performance?

**Setup**:
- Baseline: Train on all complexities from start
- Curriculum: Easy → Medium → Hard progression
- Compare: final F1, training time, stability

### Experiment 3: Soft Subtraction Tuning
**Question**: What's the optimal subtraction strength?

**Setup**:
- Vary soft_subtract_alpha from 0.5 to 1.0
- Test on overlapping track events
- Optimize: track separation quality vs. residual preservation

---

## Quick Reference Commands

### Check job status
```bash
squeue -u am4qw                    # Your jobs
squeue -u am4qw --start            # Estimated start times
scancel <job_id>                   # Cancel a job
```

### Monitor training
```bash
tail -f logs/<job_id>.out          # Live output
grep "epoch" logs/<job_id>.out     # Training progress
```

### Submit visualization
```bash
cd /scratch/am4qw/Qtracker_basic/QTracker_training
./scripts/submit_viz.sh --event 0 --type single --format mp4
./scripts/submit_viz.sh --event 5 --type multi --format gif
```

### Copy results back to local Mac
```bash
# On Mac
scp -r am4qw@rivanna.hpc.virginia.edu:/scratch/am4qw/Qtracker_basic/QTracker_training/plots/animations/ ~/Downloads/
```

---

## Troubleshooting

### "No module named 'matplotlib'"
**Solution**: You're not in the container. Use SLURM scripts which handle containers automatically.

### "Invalid account or account/partition combination"
**Solution**: Check available accounts with `sacctmgr show user am4qw -s`

### "CUDA out of memory"
**Solution**: Reduce batch size in training config or request more GPUs

### "File not found: checkpoints/model.keras"
**Solution**: Check if model path is correct. Donghwa's models are in `/project/ptgroup/spinquest/Donghwa/...`

---

## Contact & Resources

- **Project GitHub**: https://github.com/uva-spin/Qtracker_basic
- **Rivanna Documentation**: https://www.rc.virginia.edu/userinfo/rivanna/overview/
- **SLURM Guide**: https://www.rc.virginia.edu/userinfo/rivanna/slurm/
- **Your branch**: `AnveshTrackFinderWork`
