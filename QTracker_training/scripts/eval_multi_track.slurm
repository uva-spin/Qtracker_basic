#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -J eval_multi_track
#SBATCH -o Slurm_Files/eval_multi_track.out
#SBATCH -e Slurm_Files/eval_multi_track.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif

# Override any of these at submission time, e.g.:
#   CODEDIR=... DATADIR=... MODEL=... sbatch scripts/eval_multi_track.slurm
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
_DEFAULT_DATADIR=/project/ptgroup/spinquest/Anvesh
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-${_DEFAULT_DATADIR}}
DATA=${DATA:-${DATADIR}/data/multi_track/processed_files/mc_events_val.root}
MODEL=${MODEL:-/mnt/code/checkpoints/multi_track_finder_best.keras}
MAX_PAIRS=${MAX_PAIRS:-3}

### Main Body ###
# --- Evaluating Multi-Track Finder ---
EVAL_ARGS="--max_pairs ${MAX_PAIRS}"
[ -n "${NUM_EVENTS}" ] && EVAL_ARGS="${EVAL_ARGS} --num_events ${NUM_EVENTS}"

${APPTAINER} exec --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env PYTHONUNBUFFERED=1 \
    --bind ${CODEDIR}:/mnt/code \
    --bind ${DATADIR}:/mnt/data \
    "${IMAGE}" \
    python3 -u /mnt/code/eval_multi_track.py \
        /mnt/data/data/multi_track/processed_files/mc_events_val.root \
        ${MODEL} \
        ${EVAL_ARGS} \
        2>&1 | tee /mnt/code/results/multi_track_results.txt
