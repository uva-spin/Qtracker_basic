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

echo "=== TF GPU check (verbose) ==="
${APPTAINER} exec --nv --env TF_FORCE_GPU_ALLOW_GROWTH=true --env TF_CPP_MIN_LOG_LEVEL=0 "${IMAGE}" \
    python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
print('TF version:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"

echo "=== LD_DEBUG check for missing libs ==="
${APPTAINER} exec --nv "${IMAGE}" \
    python3 -c "
import ctypes, sys
libs = ['libcuda.so.1', 'libcudart.so', 'libcudnn.so', 'libcublas.so']
for lib in libs:
    try:
        ctypes.CDLL(lib)
        print(f'OK: {lib}')
    except OSError as e:
        print(f'MISSING: {lib} -> {e}')
"
