#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J fixed-track-finder
#SBATCH -o Slurm_Files/fixed-track-finder.out
#SBATCH -e Slurm_Files/fixed-track-finder.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/


### Main Body ###
# --- Training and Evaluating Fixed-N Track Finder ---
BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0
N=${N_PAIRS:-1}  # default to N=1; override with: sbatch --export=N_PAIRS=2 train_finder.slurm

# GPU validation
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    echo $CUDA_VISIBLE_DEVICES

${APPTAINER} exec --nv --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/MultiTrackFinder.py \
     /mnt/code/data/multi_track/processed_files/mc_events_train_low_N${N}.root \
     /mnt/code/data/multi_track/processed_files/mc_events_val_N${N}.root \
     --train_root_file_med /mnt/code/data/multi_track/processed_files/mc_events_train_med_N${N}.root \
     --train_root_file_high /mnt/code/data/multi_track/processed_files/mc_events_train_high_N${N}.root \
     --output_model /mnt/code/checkpoints/finder_N${N}.keras \
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
     --batch_size 64 \
     --dropout_bn 0.5 \
     --dropout_enc 0.4 \
     --dropout_attn 0.1 \
     --factor 0.3 \
     --pos_weight 20.0 \
     --n_pairs $N

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/eval_multi_track.py \
     /mnt/code/data/multi_track/processed_files/mc_events_val_N${N}.root \
     /mnt/code/checkpoints/finder_N${N}.keras \
     --n_pairs $N
