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
LEARNING_RATE=0.0003
BATCH_NORM=1

# # Denoiser - best performing (pos_weight 15.0)
# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/TrackFinder_denoise.py \
#      /mnt/code/data/processed_files/mc_events_train_denoise_low.root \
#      /mnt/code/data/processed_files/mc_events_train_denoise_med.root \
#      /mnt/code/data/processed_files/mc_events_train_denoise_high.root \
#      /mnt/code/data/processed_files/mc_events_val_denoise.root \
#      --output_model /mnt/code/checkpoints/track_finder_denoise.h5 \
#      --learning_rate $LEARNING_RATE \
#      --patience 12 \
#      --batch_norm $BATCH_NORM \
#      --base 64 \
#      --epochs 60 \
#      --batch_size 64 \
#      --dropout_bn 0.5 \
#      --dropout_enc 0.4 \
#      --pos_weight 15.0

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate_denoise.py

# # Segmenter
# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/TrackFinder_unetpp_seg.py \
#      /mnt/code/data/processed_files/mc_events_train_denoise_low.root \
#      /mnt/code/data/processed_files/mc_events_train_denoise_med.root \
#      /mnt/code/data/processed_files/mc_events_train_denoise_high.root \
#      /mnt/code/data/processed_files/mc_events_val_seg.root \
#      --output_model /mnt/code/checkpoints/track_finder_seg.h5 \
#      --learning_rate $LEARNING_RATE \
#      --patience 12 \
#      --batch_norm $BATCH_NORM \
#      --base 64 \
#      --epochs 60 \
#      --batch_size 64 \
#      --dropout_bn 0.5 \
#      --dropout_enc 0.4 \
#      --denoise_model_path /mnt/code/checkpoints/track_finder_denoise.h5

# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/evaluate_v2.py \
#      /mnt/code/data/processed_files/mc_events_val_denoise.root \
#      /mnt/code/checkpoints/track_finder_denoise.h5 \
#      /mnt/code/checkpoints/track_finder_seg.h5

# Curriculum learning in denoiser

# +) Also test different learning rates for each curriculum stage,
#    cosine annealing LR (with and without warmup)

    
## Best Performing ##
# ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 /mnt/code/models/TrackFinder_unetpp.py \
#      /mnt/code/data/processed_files/mc_events_train_low.root \
#      /mnt/code/data/processed_files/mc_events_train_med.root \
#      /mnt/code/data/processed_files/mc_events_train_high.root \
#      /mnt/code/data/processed_files/mc_events_val.root \
#      --output_model /mnt/code/checkpoints/track_finder_unetpp1.h5 \
#      --learning_rate $LEARNING_RATE \
#      --factor 0.3 \
#      --min_lr 0.000001 \
#      --patience 12 \
#      --batch_norm $BATCH_NORM \
#      --base 64 \
#      --epochs 60 \
#      --batch_size 64 \
#      --dropout_bn 0.5 \
#      --dropout_enc 0.4 \
#      --low_ratio 0.5 \
#      --med_ratio 0.8


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
