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
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=path/to/Qtracker_basic/QTracker_prod


### Main Body ###
# --- Training and Evaluating Track Finder ---
LEARNING_RATE=0.00005

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/training_scripts/TrackFinder_prod.py \
     /mnt/code/data/processed_files/mc_events.root \
    --output_model models/track_finder.h5 \
    --learning_rate $LEARNING_RATE

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events.root \
     /mnt/code/models/track_finder.h5

# --- Training and Evaluating Momentum Model ---
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

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/QTracker_prod.py \
     /mnt/code/data/raw_files/${QTRACKER_INPUT}.root \
     --output_file /mnt/code/data/processed_files/qtracker_reco.root
