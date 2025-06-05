#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J tunertest
#SBATCH -o Slurm_Files/tuner.out
#SBATCH -e Slurm_Files/tuner.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0
source /project/ptgroup/spinquest/this-e1039.sh

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif \
    training_scripts/TrackFinder_prod.py \
    data/noisy_output.root
