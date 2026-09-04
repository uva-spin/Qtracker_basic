#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J diagnose-localization
#SBATCH -o Slurm_Files/diagnose-localization.out
#SBATCH -e Slurm_Files/diagnose-localization.err
#SBATCH --mem=64000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

# Override at submission time, e.g.:
#   MODEL=/mnt/data/checkpoints/multi_track_finder_best.keras sbatch scripts/diagnose_localization.slurm
MODEL=${MODEL:-/mnt/data/checkpoints/multi_track_finder_best.keras}
MAX_PAIRS=${MAX_PAIRS:-3}
NUM_EVENTS=${NUM_EVENTS:-5000}

${APPTAINER} exec --nv --pwd /mnt/code --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/diagnose_localization.py \
     ${MODEL} \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --max_pairs ${MAX_PAIRS} \
     --num_events ${NUM_EVENTS}
