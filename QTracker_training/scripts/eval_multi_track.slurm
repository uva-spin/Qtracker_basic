#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -J eval_multi_track
#SBATCH -o Slurm_Files/eval_multi_track.out
#SBATCH -e Slurm_Files/eval_multi_track.err
#SBATCH --mem=256000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=${CODEDIR:-/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training}

DATA=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/data/processed_files/mc_events_val.root
MODEL=${MODEL:-/mnt/code/checkpoints/multi_track_finder.keras}

### Main Body ###
# --- Evaluating Multi-Track Finder ---
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/eval_multi_track.py \
        ${DATA} \
        ${MODEL} \
        2>&1 | tee ${CODEDIR}/results/multi_track_results.txt
