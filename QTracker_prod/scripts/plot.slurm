#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J plottest
#SBATCH -o Slurm_Files/plot.out
#SBATCH -e Slurm_Files/plot.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR="/home/gcs3ja/home/Qtracker_basic/QTracker_prod"


### Main Body ###
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate_denoise.py \
     --val_file_path /mnt/code/data/processed_files/mc_events_val.root \
     --model_path /mnt/code/checkpoints/track_finder_denoise.h5