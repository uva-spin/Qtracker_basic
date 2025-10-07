#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
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
BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0

# Denoise-base 16 and 32 are not too different in performance
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/TrackFinder_joint.py \
     /mnt/code/data/processed_files/mc_events_train_low.root \
     /mnt/code/data/processed_files/mc_events_train_med.root \
     /mnt/code/data/processed_files/mc_events_train_high.root \
     /mnt/code/data/processed_files/mc_events_val.root \
     --output_model /mnt/code/checkpoints/track_finder_joint.keras \
     --lr_low 0.0003 \
     --lr_med 0.0001 \
     --lr_high 0.00003 \
     --patience 10 \
     --lr_patience 3 \
     --batch_norm $BATCH_NORM \
     --use_attn $USE_ATTENTION \
     --use_attn_ffn $USE_ATTENTION_FFN \
     --denoise_base 32 \
     --base 64 \
     --epochs 60 \
     --batch_size 32 \
     --dropout_bn 0.5 \
     --dropout_enc 0.4 \
     --dropout_attn 0.1 \
     --factor 0.3 \
     --pos_weight 20.0

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/track_finder_joint.keras

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
