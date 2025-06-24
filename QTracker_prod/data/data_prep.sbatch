#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J Realrun
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v3.sif
CODEDIR=/scratch/bab6cw/Qtracker_basic/QTracker_prod/data
  
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/separate.py /mnt/code/JPsi_Dump_50K.root
  
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/gen_training.py /mnt/code/JPsi_Dump_50K_track1.root /mnt/code/JPsi_Dump_50K_track2.root