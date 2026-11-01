#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J single-track-finder-v2
#SBATCH -o Slurm_Files/single-track-finder-v2.out
#SBATCH -e Slurm_Files/single-track-finder-v2.err
#SBATCH --mem=256000
#SBATCH --export=CODEDIR,DATADIR

module purge

# Same as scripts/train.slurm, but:
# - trained under the v2 image (TfRootBuild_v2.sif) instead of v1, so the
#   resulting checkpoint is loadable by whichever image is current -- this is
#   a from-scratch replacement for track_finder_64.keras, whose older-Keras
#   internals (keras.src.engine.functional) neither v1 nor v2 can deserialize
#   any more.
# - writes the checkpoint to DATADIR (persistent project storage), not into
#   the scratch clone, since /scratch/am4qw/Qtracker_basic recurringly
#   corrupts and a 72hr training run isn't worth risking on that.
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif

_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
_DEFAULT_DATADIR=/project/ptgroup/spinquest/Anvesh
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-${_DEFAULT_DATADIR}}

### Main Body ###
# --- Training and Evaluating Track Finder ---
BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0

mkdir -p "${DATADIR}/checkpoints"
OUTPUT_MODEL=/mnt/data/checkpoints/track_finder_64_v2.keras

# GPU validation
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    echo $CUDA_VISIBLE_DEVICES

# Denoise-base 16 and 32 are not too different in performance
${APPTAINER} exec --nv --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/TrackFinder.py \
     /mnt/code/data/processed_files/mc_events_train_low.root \
     /mnt/code/data/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/code/data/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/code/data/processed_files/mc_events_train_high.root \
     --output_model ${OUTPUT_MODEL} \
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
     --pos_weight 20.0

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/evaluate.py \
     /mnt/code/data/processed_files/mc_events_val.root \
     ${OUTPUT_MODEL}
