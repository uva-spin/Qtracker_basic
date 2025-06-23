#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J tunertest
#SBATCH -o Slurm_Files/tuner.out
#SBATCH -e Slurm_Files/tuner.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=path/to/TfRootBuild.sif
CODEDIR=path/to/Qtracker_basic/QTracker_prod


### Main Body ###
# # --- Install Dependencies ---
# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     pip install --user tensorflow==2.15.0 uproot numba

# --- Training Track Finder ---
LEARNING_RATE=0.00005

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/training_scripts/TrackFinder_prod.py \
     /mnt/code/data/processed_files/finder_training.root \
    --output_model models/track_finder.h5 \
    --learning_rate $LEARNING_RATE

# --- Training Momentum Model ---
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/training_scripts/Momentum_training.py \
     /mnt/code/data/processed_files/momentum_training-1.root \
    --output models/mom_mup.h5 \
    --epochs 300 \
    --batch_size 32

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/training_scripts/Momentum_training.py \
     /mnt/code/data/processed_files/momentum_training-2.root \
    --output models/mom_mum.h5 \
    --epochs 300 \
    --batch_size 32
