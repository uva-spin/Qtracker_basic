#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 18:00:00
#SBATCH -J autoregressive-eval
#SBATCH -o Slurm_Files/autoregressive-eval.out
#SBATCH -e Slurm_Files/autoregressive-eval.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
# Two different images for the two steps below, not one for the whole job:
# - IMAGE_INFER (v1, Donghwa's original image): track_finder_64.keras was
#   saved under v1's older Keras internals, and v2's Keras can't deserialize
#   it (TypeError: Could not deserialize class 'Functional' because its
#   parent module keras.src.engine.functional cannot be imported).
# - IMAGE_EVAL (v2): eval_autoregressive.py imports QTracker.py, which
#   unconditionally does `from numba import njit, prange` -- numba isn't
#   installed in v1 (ModuleNotFoundError there), but every v2-based job this
#   project has run imports QTracker fine.
# They only communicate through the .npz file in between, which doesn't care
# which image wrote/read it. If a checkpoint trained/saved under v2 is used
# for MODEL instead, IMAGE_INFER can switch to v2 too.
IMAGE_INFER=/project/ptgroup/spinquest/David/TfRootBuild.sif
IMAGE_EVAL=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif

# Override any of these at submission time, e.g.:
#   CODEDIR=... DATADIR=... MODEL=... sbatch scripts/eval_autoregressive.slurm
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
_DEFAULT_DATADIR=/project/ptgroup/spinquest/Anvesh
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-${_DEFAULT_DATADIR}}

### Main Body ###
MAX_PAIRS=3
THRESHOLDS="0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# Single-track model used as the base finder -- a container-side path under
# /mnt/data (i.e. relative to DATADIR), NOT a project-shared default. Point
# this at whichever trained single-track TrackFinder checkpoint you want to
# test (e.g. the one behind the 42.2%/42.7% baseline in CLAUDE.md).
MODEL=${MODEL:?Set MODEL to a single-track TrackFinder checkpoint inside /mnt/data, e.g. MODEL=/mnt/data/checkpoints/track_finder_64.keras}

# Multi-track validation data
VAL_DATA=/mnt/data/data/multi_track/processed_files/mc_events_val.root

RUN_NAME=ar_eval_${SLURM_JOB_ID}
OUTPUT_DIR=/mnt/data/checkpoints/autoregressive_eval/${RUN_NAME}
mkdir -p "${DATADIR}/checkpoints/autoregressive_eval/${RUN_NAME}"

for THRESH in $THRESHOLDS; do
    echo ""
    echo "######################################################################"
    echo "# THRESHOLD = ${THRESH}"
    echo "######################################################################"
    echo ""

    OUTPUT=${OUTPUT_DIR}/autoregressive_predictions_${THRESH}.npz

    # Step 1: Run autoregressive prediction
    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE_INFER}" \
        python3 /mnt/code/models/AutoregressiveTrackFinder.py \
         ${VAL_DATA} \
         ${MODEL} \
         --max_pairs $MAX_PAIRS \
         --confidence_threshold $THRESH \
         --output ${OUTPUT}

    # Step 2: Evaluate
    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE_EVAL}" \
        python3 /mnt/code/eval_autoregressive.py ${OUTPUT} ${VAL_DATA}
done
