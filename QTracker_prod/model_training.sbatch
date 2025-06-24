#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J Realrun
#SBATCH -o Slurm_Files/data.out
#SBATCH -e Slurm_Files/data.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v3.sif
CODEDIR=/scratch/bab6cw/Qtracker_basic/QTracker_prod

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/training_scripts/TrackFinder_prod.py /mnt/code/data/finder_training_Dump_100K.root --output=/mnt/code/models/track_finder_Dump_100K.h5

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/training_scripts/Momentum_training.py /mnt/code/data/momentum_training-1_Dump_100K.root --output=/mnt/code/models/mom_mup_Dump_100K.h5

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/training_scripts/Momentum_training.py /mnt/code/data/momentum_training-2_Dump_100K.root --output=/mnt/code/models/mom_mum_Dump_100K.h5

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/QTracker_prod.py data/JPsi_Dump_100K.root
  
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
  python3 /mnt/code/Util/imass_plot.py qtracker_reco.root