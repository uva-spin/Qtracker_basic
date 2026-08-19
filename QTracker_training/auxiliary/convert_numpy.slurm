#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -J root-to-numpy
#SBATCH -o Slurm_Files/root_to_numpy.out
#SBATCH -e Slurm_Files/root_to_numpy.err
#SBATCH --mem=64000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=${CODEDIR:-/scratch/am4qw/Qtracker_basic/QTracker_training}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

${APPTAINER} exec \
    --bind ${CODEDIR}:/mnt/code \
    --bind ${DATADIR}:/mnt/data \
    "${IMAGE}" \
    python3 /mnt/code/data/root_to_numpy.py \
    --data_dir /mnt/data/data/multi_track/processed_files \
    --out_dir  /mnt/data/data/multi_track/numpy_files \
    --max_pairs 5
