#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -J fno-denoiser
#SBATCH -o Slurm_Files/fno-denoiser.out
#SBATCH -e Slurm_Files/fno-denoiser.err
#SBATCH --mem=128000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

### Main Body ###
RUN_TAG=fno_denoiser_v1
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=/mnt/data/checkpoints/fno_denoiser/${RUN_NAME}

${APPTAINER} exec --nv --pwd /mnt/code --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/experiments/fno_denoiser/train.py \
     /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
     --output_model ${OUTPUT_DIR}/fno_denoiser.keras \
     --max_pairs 3 \
     --base 32 \
     --fno_depth 4 \
     --k_max 32 \
     --num_heads 4 \
     --pos_weight 20.0 \
     --epochs 60 \
     --med_ratio 0.7 \
     --batch_size 32 \
     --patience 12 \
     --mlflow_experiment fno_denoiser \
     --mlflow_run_name "${RUN_NAME}"
