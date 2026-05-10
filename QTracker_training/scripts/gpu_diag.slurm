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
nvidia-smi | head -10

echo "=== Container v1 (current) ==="
${APPTAINER} exec --nv /project/ptgroup/spinquest/David/TfRootBuild.sif \
    python3 -c "import tensorflow as tf; bi=tf.sysconfig.get_build_info(); print('TF:', tf.__version__, 'CUDA:', bi.get('cuda_version','?')); print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>&1 | grep -E "TF:|GPUs:|dlopen|CUDA_ERROR"

echo "=== Container v2 ==="
${APPTAINER} exec --nv /project/ptgroup/spinquest/David/TfRootBuild_v2.sif \
    python3 -c "import tensorflow as tf; bi=tf.sysconfig.get_build_info(); print('TF:', tf.__version__, 'CUDA:', bi.get('cuda_version','?')); print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>&1 | grep -E "TF:|GPUs:|dlopen|CUDA_ERROR"

echo "=== Container v3 ==="
${APPTAINER} exec --nv /project/ptgroup/spinquest/David/TfRootBuild_v3.sif \
    python3 -c "import tensorflow as tf; bi=tf.sysconfig.get_build_info(); print('TF:', tf.__version__, 'CUDA:', bi.get('cuda_version','?')); print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>&1 | grep -E "TF:|GPUs:|dlopen|CUDA_ERROR"
