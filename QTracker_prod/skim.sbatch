#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J Realrun
#SBATCH -o Slurm_Files/skim.out
#SBATCH -e Slurm_Files/skim.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v3.sif
CODEDIR=/scratch/bab6cw/Qtracker_basic/QTracker_prod

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/Util/skim.py /mnt/code/data/JPsi_Dump_100K.root --output_file=/mnt/code/data/JPsi_Dump_50K.root --max_events=50000
  