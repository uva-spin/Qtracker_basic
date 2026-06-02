#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -J qmetric-components-v2
#SBATCH -o Slurm_Files/qmetric-components-v2-%j.out
#SBATCH -e Slurm_Files/qmetric-components-v2-%j.err
#SBATCH --mem=64000
#SBATCH --export=ALL

set -euo pipefail
module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/scratch/bab6cw/QTracker_training

INPUT_CSV=${INPUT_CSV:-/mnt/code/outputs/qmetric_single_val_arrays.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/code/outputs/qmetric_component_comparison_v2_single_val}
THRESHOLDS=${THRESHOLDS:-0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,0.995}
KEEP_FRACTIONS=${KEEP_FRACTIONS:-0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.25,0.20,0.10}
MUP_MODEL=${MUP_MODEL:-/mnt/code/checkpoints/mom_mup.h5}
MUM_MODEL=${MUM_MODEL:-/mnt/code/checkpoints/mom_mum.h5}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
APPLY_MOMENTUM_MASK=${APPLY_MOMENTUM_MASK:-1}
WRITE_CANDIDATE_PREDICTIONS=${WRITE_CANDIDATE_PREDICTIONS:-0}
PLOTS=${PLOTS:-1}
STRIP_USER_SITE=${STRIP_USER_SITE:-0}

mkdir -p ${CODEDIR}/Slurm_Files
mkdir -p ${CODEDIR}/outputs

printf "=== QMetric component comparison v2 job ===\n"
printf "APPTAINER=%s\n" "${APPTAINER}"
printf "IMAGE=%s\n" "${IMAGE}"
printf "CODEDIR=%s\n" "${CODEDIR}"
printf "INPUT_CSV=%s\n" "${INPUT_CSV}"
printf "OUTPUT_DIR=%s\n" "${OUTPUT_DIR}"
printf "THRESHOLDS=%s\n" "${THRESHOLDS}"
printf "KEEP_FRACTIONS=%s\n" "${KEEP_FRACTIONS}"
printf "MUP_MODEL=%s\n" "${MUP_MODEL}"
printf "MUM_MODEL=%s\n" "${MUM_MODEL}"
printf "CHUNK_SIZE=%s\n" "${CHUNK_SIZE}"
printf "APPLY_MOMENTUM_MASK=%s\n" "${APPLY_MOMENTUM_MASK}"
printf "WRITE_CANDIDATE_PREDICTIONS=%s\n" "${WRITE_CANDIDATE_PREDICTIONS}"
printf "PLOTS=%s\n" "${PLOTS}"
printf "STRIP_USER_SITE=%s\n" "${STRIP_USER_SITE}"
printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-unset}"
printf "=========================================\n"

if [ ! -f "${CODEDIR}/scripts/compare_qmetric_components_v2.py" ]; then
    echo "[fatal] Missing ${CODEDIR}/scripts/compare_qmetric_components_v2.py"
    exit 1
fi

${APPTAINER} exec --nv \
    --bind ${CODEDIR}:/mnt/code \
    ${IMAGE} \
    bash -lc "
        export EXTRA_NVIDIA_LIBS=\$(python3 - <<'PY'
import glob
import os
import site

paths = []
for base in site.getsitepackages() + [site.getusersitepackages()]:
    paths.extend(glob.glob(os.path.join(base, 'nvidia', '*', 'lib')))
paths = [p for p in paths if os.path.isdir(p)]
print(':'.join(paths))
PY
)
        export LD_LIBRARY_PATH=\${EXTRA_NVIDIA_LIBS}:\${LD_LIBRARY_PATH}
        export KERAS_BACKEND=tensorflow
        export TF_CPP_MIN_LOG_LEVEL=3
        export STRIP_USER_SITE=${STRIP_USER_SITE}

        echo \"EXTRA_NVIDIA_LIBS=\${EXTRA_NVIDIA_LIBS}\"
        python3 - <<'PY'
import tensorflow as tf
print('[gpu check inside run] physical GPUs:', tf.config.list_physical_devices('GPU'))
PY

        python3 /mnt/code/scripts/compare_qmetric_components_v2.py \
            ${INPUT_CSV} \
            --output_dir ${OUTPUT_DIR} \
            --thresholds ${THRESHOLDS} \
            --keep_fractions ${KEEP_FRACTIONS} \
            --mup_model ${MUP_MODEL} \
            --mum_model ${MUM_MODEL} \
            --chunk_size ${CHUNK_SIZE} \
            --apply_momentum_mask ${APPLY_MOMENTUM_MASK} \
            --write_candidate_predictions ${WRITE_CANDIDATE_PREDICTIONS} \
            --plots ${PLOTS} \
            --print_env 1
    "

echo "[done] Component comparison v2 written to ${OUTPUT_DIR}"
