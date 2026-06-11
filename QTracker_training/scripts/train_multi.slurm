#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J multi-track-finder
#SBATCH -o Slurm_Files/multi-track-finder.out
#SBATCH -e Slurm_Files/multi-track-finder.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
# Anvesh's personal project storage: owns the data, persists mlruns
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}


### Main Body ###
BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0
MAX_PAIRS=3

${APPTAINER} exec --nv --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/MultiTrackFinder.py \
     /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
     --output_model /mnt/code/checkpoints/multi_track_finder.keras \
     --lr_low 0.0003 \
     --lr_med 0.0001 \
     --lr_high 0.00003 \
     --patience 20 \
     --lr_patience 5 \
     --batch_norm $BATCH_NORM \
     --use_attn $USE_ATTENTION \
     --use_attn_ffn $USE_ATTENTION_FFN \
     --denoise_base 32 \
     --base 64 \
     --epochs 80 \
     --batch_size 32 \
     --dropout_bn 0.5 \
     --dropout_enc 0.2 \
     --dropout_attn 0.1 \
     --factor 0.5 \
     --pos_weight 20.0 \
     --max_pairs $MAX_PAIRS \
     --mlflow_experiment multi_track_v2 \
     --mlflow_run_name "slurm_${SLURM_JOB_ID}"

${APPTAINER} exec --nv --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/eval_multi_track.py \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/multi_track_finder.keras \
     --max_pairs $MAX_PAIRS
