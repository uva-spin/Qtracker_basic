#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -J autoregressive-eval
#SBATCH -o Slurm_Files/autoregressive-eval.out
#SBATCH -e Slurm_Files/autoregressive-eval.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

### Main Body ###
MAX_ITERS=5
CONFIDENCE_THRESHOLD=0.15

# Single-track model used as the base finder
MODEL=/mnt/code/checkpoints/track_finder_64.keras

# Multi-track validation data
VAL_DATA=/mnt/code/data/multi_track/processed_files/mc_events_val.root

# --- Step 1: Run autoregressive prediction ---
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/AutoregressiveTrackFinder.py \
     ${VAL_DATA} \
     ${MODEL} \
     --max_iters $MAX_ITERS \
     --confidence_threshold $CONFIDENCE_THRESHOLD \
     --max_pairs 5 \
     --output /mnt/code/checkpoints/autoregressive_predictions.npz

# --- Step 2: Evaluate ---
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/eval_autoregressive.py \
     /mnt/code/checkpoints/autoregressive_predictions.npz
