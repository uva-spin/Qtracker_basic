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
module load cuda/12.2.2

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif

# Override any of these at submission time, e.g.:
#   CODEDIR=... DATADIR=... MODEL=... sbatch scripts/eval_multi_track.slurm
_DEFAULT_BASE=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_BASE}}
DATADIR=${DATADIR:-${_DEFAULT_BASE}}
DATA=${DATA:-${DATADIR}/data/processed_files/mc_events_val.root}
MODEL=${MODEL:-${DATADIR}/checkpoints/multi_track_finder.keras}

### Main Body ###
# --- Evaluating Multi-Track Finder ---
${APPTAINER} exec --nv \
    --bind ${CODEDIR}:/mnt/code \
    --bind ${DATADIR}:/mnt/data \
    "${IMAGE}" \
    python3 /mnt/code/eval_multi_track.py \
        ${DATA} \
        ${MODEL} \
        2>&1 | tee ${CODEDIR}/results/multi_track_results.txt
