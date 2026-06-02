#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -J qmetric-pattern
#SBATCH -o Slurm_Files/qmetric-pattern-%j.out
#SBATCH -e Slurm_Files/qmetric-pattern-%j.err
#SBATCH --mem=64000
#SBATCH --export=ALL

set -euo pipefail
module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/scratch/bab6cw/QTracker_training

ROOT_FILE=${ROOT_FILE:-/mnt/code/data/processed_files/mc_events_val.root}
MODEL_PATH=${MODEL_PATH:-/mnt/code/checkpoints/single_track_finder.keras}
OUTPUT=${OUTPUT:-/mnt/code/outputs/qmetric_single_val_arrays.csv}
MULTI_TRACK=${MULTI_TRACK:-0}
MAX_PAIRS=${MAX_PAIRS:-5}
USE_DENOISE_LOADER=${USE_DENOISE_LOADER:-0}
CHUNK_SIZE=${CHUNK_SIZE:-128}
LOCAL_RADIUS=${LOCAL_RADIUS:-2}
MAX_EVENTS=${MAX_EVENTS:-0}
MODEL_KIND=${MODEL_KIND:-single}
LOAD_MODE=${LOAD_MODE:-rebuild_single}
STRIP_USER_SITE=${STRIP_USER_SITE:-0}
SAVE_CANDIDATE_ARRAYS=${SAVE_CANDIDATE_ARRAYS:-1}
SAVE_TRUTH_MOMENTUM=${SAVE_TRUTH_MOMENTUM:-1}
SNAP_TO_REAL_HITS=${SNAP_TO_REAL_HITS:-1}
DETECTOR_ID_BASE=${DETECTOR_ID_BASE:-auto}

BATCH_NORM=${BATCH_NORM:-1}
USE_ATTN=${USE_ATTN:-1}
USE_ATTN_FFN=${USE_ATTN_FFN:-0}
DENOISE_BASE=${DENOISE_BASE:-32}
BASE=${BASE:-64}
DROPOUT_BN=${DROPOUT_BN:-0.5}
DROPOUT_ENC=${DROPOUT_ENC:-0.4}
DROPOUT_ATTN=${DROPOUT_ATTN:-0.1}
DEEP_SUPERVISION=${DEEP_SUPERVISION:-0}

mkdir -p ${CODEDIR}/Slurm_Files
mkdir -p ${CODEDIR}/outputs

printf "=== QMetric pattern table job ===\n"
printf "APPTAINER=%s\n" "${APPTAINER}"
printf "IMAGE=%s\n" "${IMAGE}"
printf "CODEDIR=%s\n" "${CODEDIR}"
printf "ROOT_FILE=%s\n" "${ROOT_FILE}"
printf "MODEL_PATH=%s\n" "${MODEL_PATH}"
printf "OUTPUT=%s\n" "${OUTPUT}"
printf "MULTI_TRACK=%s\n" "${MULTI_TRACK}"
printf "MAX_PAIRS=%s\n" "${MAX_PAIRS}"
printf "USE_DENOISE_LOADER=%s\n" "${USE_DENOISE_LOADER}"
printf "CHUNK_SIZE=%s\n" "${CHUNK_SIZE}"
printf "MAX_EVENTS=%s\n" "${MAX_EVENTS}"
printf "MODEL_KIND=%s\n" "${MODEL_KIND}"
printf "LOAD_MODE=%s\n" "${LOAD_MODE}"
printf "SAVE_CANDIDATE_ARRAYS=%s\n" "${SAVE_CANDIDATE_ARRAYS}"
printf "SAVE_TRUTH_MOMENTUM=%s\n" "${SAVE_TRUTH_MOMENTUM}"
printf "SNAP_TO_REAL_HITS=%s\n" "${SNAP_TO_REAL_HITS}"
printf "DETECTOR_ID_BASE=%s\n" "${DETECTOR_ID_BASE}"
printf "STRIP_USER_SITE=%s\n" "${STRIP_USER_SITE}"
printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-unset}"
printf "=================================\n"


echo "[run] Building QMetric pattern table with candidate arrays"
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

        python3 /mnt/code/scripts/build_quality_pattern_table_arrays.py \
            ${ROOT_FILE} \
            ${MODEL_PATH} \
            --output ${OUTPUT} \
            --multi_track ${MULTI_TRACK} \
            --max_pairs ${MAX_PAIRS} \
            --use_denoise_loader ${USE_DENOISE_LOADER} \
            --chunk_size ${CHUNK_SIZE} \
            --local_radius ${LOCAL_RADIUS} \
            --max_events ${MAX_EVENTS} \
            --model_kind ${MODEL_KIND} \
            --load_mode ${LOAD_MODE} \
            --save_candidate_arrays ${SAVE_CANDIDATE_ARRAYS} \
            --save_truth_momentum ${SAVE_TRUTH_MOMENTUM} \
            --snap_to_real_hits ${SNAP_TO_REAL_HITS} \
            --detector_id_base ${DETECTOR_ID_BASE} \
            --batch_norm ${BATCH_NORM} \
            --use_attn ${USE_ATTN} \
            --use_attn_ffn ${USE_ATTN_FFN} \
            --denoise_base ${DENOISE_BASE} \
            --base ${BASE} \
            --dropout_bn ${DROPOUT_BN} \
            --dropout_enc ${DROPOUT_ENC} \
            --dropout_attn ${DROPOUT_ATTN} \
            --deep_supervision ${DEEP_SUPERVISION} \
            --print_env 1
    "
echo "[done] Output written to ${OUTPUT}"
