#!/bin/bash
# Run this from your LOCAL Mac terminal (not SSH'd to Rivanna)
# Usage: bash scripts/pull_results.sh

REMOTE=am4qw@login.hpc.virginia.edu
REMOTE_DIR=/scratch/am4qw/Qtracker_basic/QTracker_training
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Pulling results from Rivanna -> $LOCAL_DIR"

# SLURM logs
rsync -av "$REMOTE:$REMOTE_DIR/Slurm_Files/" "$LOCAL_DIR/Slurm_Files/"

# Checkpoints
rsync -av --exclude="*.keras" "$REMOTE:$REMOTE_DIR/checkpoints/" "$LOCAL_DIR/checkpoints/"

# Plots (loss curves etc saved alongside checkpoints)
rsync -av --include="*.png" --include="*.pdf" --exclude="*" "$REMOTE:$REMOTE_DIR/checkpoints/" "$LOCAL_DIR/checkpoints/"

echo "Done."
