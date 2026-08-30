#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -J pair-count-fno
#SBATCH -o Slurm_Files/pair-count-fno.out
#SBATCH -e Slurm_Files/pair-count-fno.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
# Anvesh's personal project storage: owns the data, persists mlruns
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

### Main Body ###
MAX_PAIRS=3

# Edit RUN_TAG per submission to describe what this run is testing.
RUN_TAG=max_pairs3_v2
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=/mnt/data/checkpoints/pair_count_fno/${RUN_NAME}

# Set to "--reverse_curriculum" to train high -> med -> low instead.
REVERSE_FLAG=""

${APPTAINER} exec --nv --pwd /mnt/code --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/experiments/pair_count_fno/train.py \
     /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
     --output_model ${OUTPUT_DIR}/pair_count_fno.keras \
     --max_pairs $MAX_PAIRS \
     --base 32 \
     --fno_depth 4 \
     --k_max 32 \
     --num_heads 4 \
     --epochs 60 \
     --med_ratio 0.7 \
     --batch_size 64 \
     --patience 12 \
     --mlflow_experiment pair_count_fno \
     --mlflow_run_name "${RUN_NAME}" \
     ${REVERSE_FLAG}
