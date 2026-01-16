#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J momentumtest
#SBATCH -o Slurm_Files/momentum.out
#SBATCH -e Slurm_Files/momentum.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_prod


### Main Body ###
# --- Training and Evaluating Track Finder ---
BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0

# --- Training and Evaluating Momentum Model ---
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/Momentum_training.py \
     /mnt/code/data/processed_files/momentum_training-1_train.root \
    --output checkpoints/mom_mup.h5 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0003 \
    --dropout_rate 0.3

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/Momentum_training.py \
     /mnt/code/data/processed_files/momentum_training-2_train.root \
    --output checkpoints/mom_mum.h5 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0003 \
    --dropout_rate 0.3

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/QTracker.py \
     /mnt/code/data/processed_files/mc_events_val.root \
     --output_file /mnt/code/data/processed_files/qtracker_reco.root

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/Util/imass_plot.py \
     /mnt/code/data/processed_files/qtracker_reco.root \
     --output_plot /mnt/code/plots/invariant_mass.png

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/Qmetric_training.py \
     /mnt/code/data/processed_files/qtracker_reco.root \
