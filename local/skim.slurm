#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J skim
#SBATCH -o Slurm_Files/skim.out
#SBATCH -e Slurm_Files/skim.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

echo "Generate training data for train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/multi_track/gen_training_random.py \
        /mnt/code/data/raw_files/Dimuons_Train_500K_track1.root \
        /mnt/code/data/raw_files/Dimuons_Train_500K_track2.root \
        --output /mnt/code/data/multi_track/processed_files/finder_training_train_1_0.root \
        --pairsmup 1 \
        --pairsmum 1 \
        --random

mv momentum_training-1.root /mnt/code/data/multi_track/processed_files/momentum_training-1_train.root 
mv momentum_training-2.root /mnt/code/data/multi_track/processed_files/momentum_training-2_train.root

echo "Inject low-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/multi_track/messy_gen.py \
        /mnt/code/data/multi_track/processed_files/finder_training_train_1_0.root \
        /mnt/code/data/processed_files/single_muons_train.root \
        --output /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0.root \
        --uniform_tracks 0 \
        --lower_bound 0 \
        --num_tracks 16

${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/skim.py \
        /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0.root \
        --output_file /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0_skimmed.root \
        --max_events 400000 \
        --random 1

rm /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0.root
mv /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0_skimmed.root /mnt/code/data/multi_track/processed_files/mc_events_train_low_1_0.root