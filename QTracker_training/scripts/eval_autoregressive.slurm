#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 18:00:00
#SBATCH -J autoregressive-eval
#SBATCH -o Slurm_Files/autoregressive-eval.out
#SBATCH -e Slurm_Files/autoregressive-eval.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/.worktrees/autoregressive_track_finder/QTracker_training/

### Main Body ###
MAX_PAIRS=3
THRESHOLDS="0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# Single-track model used as the base finder
MODEL=/mnt/code/checkpoints/track_finder_64.keras

# Multi-track validation data
VAL_DATA=/mnt/code/data/multi_track/processed_files/mc_events_val.root

for THRESH in $THRESHOLDS; do
    echo ""
    echo "######################################################################"
    echo "# THRESHOLD = ${THRESH}"
    echo "######################################################################"
    echo ""

    OUTPUT=/mnt/code/checkpoints/autoregressive_predictions_${THRESH}.npz

    # Step 1: Run autoregressive prediction
    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/models/AutoregressiveTrackFinder.py \
         ${VAL_DATA} \
         ${MODEL} \
         --max_pairs $MAX_PAIRS \
         --confidence_threshold $THRESH \
         --output ${OUTPUT}

    # Step 2: Evaluate
    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/eval_autoregressive.py ${OUTPUT} ${VAL_DATA}
done
