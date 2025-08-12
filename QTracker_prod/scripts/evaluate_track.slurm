#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J tunertest
#SBATCH -o Slurm_Files/tuner.out
#SBATCH -e Slurm_Files/tuner.err
#SBATCH --mem=256000
#SBATCH --export=CODEDIR

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif


### Main Body ###
rm results/func.txt

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/messy_gen.py \
        /mnt/code/data/processed_files/finder_training_val.root \
        /mnt/code/data/processed_files/single_muons_val.root \
        --output /mnt/code/data/processed_files/mc_events_val.root \
        --num_tracks 0

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/Util/skim.py \
        /mnt/code/data/processed_files/mc_events_val.root \
        --output_file /mnt/code/data/processed_files/mc_events_val.root \
        --max_events 10000

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/evaluate_track.py \
        /mnt/code/data/processed_files/mc_events_val.root \
        /mnt/code/checkpoints/track_finder_unet.h5 \
        --output /mnt/code/results/func.txt

for (( i=1; i<26; i++ )); do
    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/data/messy_gen.py \
         /mnt/code/data/processed_files/finder_training_val.root \
         /mnt/code/data/processed_files/single_muons_val.root \
         --output /mnt/code/data/processed_files/mc_events_val.root \
         --num_tracks $i

    ${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/evaluate_track.py \
         /mnt/code/data/processed_files/mc_events_val.root \
         /mnt/code/checkpoints/track_finder_unet.h5 \
         --output /mnt/code/results/func.txt
done
