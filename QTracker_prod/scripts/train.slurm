#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J tunertest
#SBATCH -o Slurm_Files/tuner.out
#SBATCH -e Slurm_Files/tuner.err
#SBATCH --mem=256000
#SBATCH --export=CODEDIR

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif


### Main Body ###
# --- Training and Evaluating Track Finder ---
LEARNING_RATE=0.00005
BATCH_NORM=1

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/TrackFinder_unetpp.py \
     /mnt/code/data/processed_files/finder_training_train.root \
     /mnt/code/data/processed_files/mc_events_val.root \
     --output_model checkpoints/track_finder_unetpp.weights.h5 \
     --learning_rate $LEARNING_RATE \
     --batch_norm $BATCH_NORM \
     --dropout_bn 0.5 \
     --dropout_enc 0.3 \
     --deep_supervision 1

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/track_finder_unetpp.weights.h5 \
     --batch_norm $BATCH_NORM

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events_val_high.root \
     /mnt/code/checkpoints/track_finder_unetpp.weights.h5 \
     --batch_norm $BATCH_NORM

# # --- Training and Evaluating Momentum Model ---
# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/Momentum_training.py \
#      /mnt/code/data/processed_files/momentum_training-1.root \
#     --output checkpoints/mom_mup.h5 \
#     --epochs 300 \
#     --batch_size 32

# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/Momentum_training.py \
#      /mnt/code/data/processed_files/momentum_training-2.root \
#     --output checkpoints/mom_mum.h5 \
#     --epochs 300 \
#     --batch_size 32

# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/QTracker_prod.py \
#      /mnt/code/data/raw_files/JPsi_Dump_Val.root \
#      --output_file /mnt/code/data/processed_files/qtracker_reco.root
