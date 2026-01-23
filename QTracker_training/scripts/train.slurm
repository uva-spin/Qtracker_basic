#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J distribute
#SBATCH -o Slurm_Files/distribute.out
#SBATCH -e Slurm_Files/distribute.err
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

# GPU validation
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    echo $CUDA_VISIBLE_DEVICES

# Denoise-base 16 and 32 are not too different in performance
${APPTAINER} exec --nv --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/TrackFinder.py \
     /mnt/code/data/processed_files/mc_events_train_low.root \
     /mnt/code/data/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/code/data/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/code/data/processed_files/mc_events_train_high.root \
     --output_model /mnt/code/checkpoints/track_finder.keras \
     --lr_low 0.0003 \
     --lr_med 0.0001 \
     --lr_high 0.00003 \
     --patience 10 \
     --lr_patience 3 \
     --batch_norm $BATCH_NORM \
     --use_attn $USE_ATTENTION \
     --use_attn_ffn $USE_ATTENTION_FFN \
     --denoise_base 16 \
     --base 32 \
     --epochs 60 \
     --batch_size 64 \
     --dropout_bn 0.5 \
     --dropout_enc 0.4 \
     --dropout_attn 0.1 \
     --factor 0.3 \
     --pos_weight 20.0

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/track_finder.keras
