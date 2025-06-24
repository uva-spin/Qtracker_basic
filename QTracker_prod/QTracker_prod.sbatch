#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J Realrun
#SBATCH -o Slurm_Files/qtr.out
#SBATCH -e Slurm_Files/qtr.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v3.sif
CODEDIR=/scratch/bab6cw/Qtracker_basic/QTracker_prod

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/QTracker_prod.py data/JPsi_Dump_80K.root
  
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/Util/imass_plot.py qtracker_reco.root