#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J Realrun
#SBATCH -o Slurm_Files/eva.out
#SBATCH -e Slurm_Files/eva.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v3.sif
CODEDIR=/scratch/bab6cw/Qtracker_basic/QTracker_prod

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/evaluate.py data/finder_training_100K.root models/track_finder.h5
  
  
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/Util/imass_plot.py qtracker_reco.root