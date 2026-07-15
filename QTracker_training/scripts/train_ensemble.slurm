#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J ensemble-track-finder
#SBATCH -o Slurm_Files/ensemble-track-finder.out
#SBATCH -e Slurm_Files/ensemble-track-finder.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

### Main Body ###
N_MODELS=3
USE_ATTENTION=1
USE_ATTENTION_FFN=0

${APPTAINER} exec --nv --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/EnsembleTrackFinder.py \
     /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
     --output_dir /mnt/code/checkpoints/ensemble \
     --n_models $N_MODELS \
     --diversity_lambda 0.05 \
     --lr_low 0.0003 \
     --lr_med 0.0001 \
     --lr_high 0.00003 \
     --patience 20 \
     --lr_patience 10 \
     --use_attn $USE_ATTENTION \
     --use_attn_ffn $USE_ATTENTION_FFN \
     --denoise_base 24 \
     --base 32 \
     --epochs 100 \
     --med_ratio 0.7 \
     --batch_size 32 \
     --dropout_bn 0.5 \
     --dropout_enc 0.2 \
     --dropout_attn 0.1 \
     --factor 0.5 \
     --pos_weight 20.0 \
     --mlflow_experiment ensemble_track_v1 \
     --mlflow_run_name "slurm_${SLURM_JOB_ID}"

${APPTAINER} exec --nv --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/eval_ensemble_multi_track.py \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/ensemble \
     --n_models $N_MODELS
