#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -t 0:10:00
#SBATCH -J gpu-diag
#SBATCH -o Slurm_Files/gpu-diag.out
#SBATCH -e Slurm_Files/gpu-diag.err
#SBATCH --mem=32000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="

echo "=== CUDA version in container ==="
${APPTAINER} exec --nv "${IMAGE}" nvcc --version 2>/dev/null || echo "nvcc not found"
${APPTAINER} exec --nv "${IMAGE}" python3 -c "import tensorflow as tf; bi=tf.sysconfig.get_build_info(); print('TF CUDA:', bi.get('cuda_version','?'), 'cuDNN:', bi.get('cudnn_version','?'))"

echo "=== TF GPU check with --nv (current) ==="
${APPTAINER} exec --nv "${IMAGE}" \
    python3 -c "import tensorflow as tf; print('GPUs (--nv):', tf.config.list_physical_devices('GPU'))"

echo "=== TF GPU check with --cleanenv + --nv ==="
${APPTAINER} exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env HOME=$HOME \
    "${IMAGE}" \
    python3 -c "import tensorflow as tf; print('GPUs (--cleanenv):', tf.config.list_physical_devices('GPU'))"

echo "=== Container CUDA lib path ==="
${APPTAINER} exec "${IMAGE}" find /usr/local -name "libcudart.so*" 2>/dev/null | head -5
