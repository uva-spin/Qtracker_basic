#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J track-counter
#SBATCH -o Slurm_Files/track-counter.out
#SBATCH -e Slurm_Files/track-counter.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/


### Main Body ###
# --- Training Track Counter ---
BATCH_NORM=1
USE_ATTENTION=1  # unused by counter but kept for CLI parity
USE_ATTENTION_FFN=0  # unused
MAX_PAIRS=3

# GPU validation
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    echo $CUDA_VISIBLE_DEVICES

${APPTAINER} exec --nv --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/TrackCounter.py \
     /mnt/code/data/multi_track/processed_files/mc_events_train_low_random.root \
     /mnt/code/data/multi_track/processed_files/mc_events_val_random.root \
     --train_root_file_med /mnt/code/data/multi_track/processed_files/mc_events_train_med_random.root \
     --train_root_file_high /mnt/code/data/multi_track/processed_files/mc_events_train_high_random.root \
     --output_model /mnt/code/checkpoints/track_counter.keras \
     --lr_low 0.0003 \
     --lr_med 0.0001 \
     --lr_high 0.00003 \
     --patience 10 \
     --lr_patience 3 \
     --batch_norm $BATCH_NORM \
     --use_attn $USE_ATTENTION \
     --denoise_base 32 \
     --epochs 60 \
     --batch_size 64 \
     --dropout_bn 0.5 \
     --dropout_enc 0.4 \
     --factor 0.3 \
     --max_pairs $MAX_PAIRS
