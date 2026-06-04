#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -J multi-track-eval
#SBATCH -o Slurm_Files/multi-track-eval.out
#SBATCH -e Slurm_Files/multi-track-eval.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

### Main Body ###
MAX_PAIRS=5

# Evaluate
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/eval_multi_track.py \
     /mnt/code/data/multi_track/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/multi_track_finder_v2.keras \
     --max_pairs $MAX_PAIRS
