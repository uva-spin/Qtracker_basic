#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J eval-ensemble
#SBATCH -o Slurm_Files/eval-ensemble.out
#SBATCH -e Slurm_Files/eval-ensemble.err
#SBATCH --mem=64000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

N_MODELS=3

${APPTAINER} exec --nv --pwd /mnt/code --env TF_FORCE_GPU_ALLOW_GROWTH=true \
    --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/eval_ensemble_multi_track.py \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/ensemble \
     --n_models $N_MODELS
