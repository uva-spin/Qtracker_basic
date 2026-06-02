#!/bin/bash
#SBATCH --job-name=qmetric_summary
#SBATCH --output=Slurm_Files/qmetric_summary_%j.out
#SBATCH --error=Slurm_Files/qmetric_summary_%j.err
#SBATCH --account=spinquest_standard
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/scratch/bab6cw/QTracker_training

INPUT_CSV=${INPUT_CSV:-/mnt/code/outputs/qmetric_single_val_debug.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/code/outputs/qmetric_summary_single_debug}
SCORE_COL=${SCORE_COL:-qmetric_score_v0}
TARGET_COL=${TARGET_COL:-mean_abs_residual_on_truth_hits}
THRESHOLDS=${THRESHOLDS:-}
QUANTILE_BINS=${QUANTILE_BINS:-4}
PLOTS=${PLOTS:-1}

mkdir -p ${CODEDIR}/Slurm_Files
mkdir -p ${CODEDIR}/outputs

echo "=== QMetric summary job ==="
echo "APPTAINER=${APPTAINER}"
echo "IMAGE=${IMAGE}"
echo "CODEDIR=${CODEDIR}"
echo "INPUT_CSV=${INPUT_CSV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SCORE_COL=${SCORE_COL}"
echo "TARGET_COL=${TARGET_COL}"
echo "THRESHOLDS=${THRESHOLDS}"
echo "QUANTILE_BINS=${QUANTILE_BINS}"
echo "PLOTS=${PLOTS}"
echo "==========================="

if [ ! -f "${CODEDIR}/scripts/summarize_quality_patterns.py" ]; then
    echo "[fatal] Missing ${CODEDIR}/scripts/summarize_quality_patterns.py"
    exit 1
fi

if [ ! -f "${IMAGE}" ]; then
    echo "[fatal] Missing container image: ${IMAGE}"
    exit 1
fi

${APPTAINER} exec \
    --bind ${CODEDIR}:/mnt/code \
    ${IMAGE} \
    bash -lc "
        export KERAS_BACKEND=tensorflow
        export TF_CPP_MIN_LOG_LEVEL=3

        echo '[check] Python / packages'
        python3 - <<'PY'
import sys
print('python:', sys.executable)
for name in ['numpy', 'pandas', 'matplotlib']:
    try:
        mod = __import__(name)
        print(name + ':', getattr(mod, '__version__', ''), getattr(mod, '__file__', ''))
    except Exception as exc:
        print(name + ' FAILED:', repr(exc))
PY

        echo '[run] Summarizing QMetric pattern table'
        python3 /mnt/code/scripts/summarize_quality_patterns.py \
            ${INPUT_CSV} \
            --output_dir ${OUTPUT_DIR} \
            --score_col ${SCORE_COL} \
            --target_col ${TARGET_COL} \
            --thresholds \"${THRESHOLDS}\" \
            --quantile_bins ${QUANTILE_BINS} \
            --plots ${PLOTS}
    "

echo "[done] Summary written to ${OUTPUT_DIR}"
