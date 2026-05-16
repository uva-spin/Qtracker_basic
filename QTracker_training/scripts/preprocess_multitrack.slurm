#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J preprocess_multitrack
#SBATCH -o Slurm_Files/preprocess_multitrack.out
#SBATCH -e Slurm_Files/preprocess_multitrack.err
#SBATCH --mem=128000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
DONGHWA=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training
OUTDIR=/project/ptgroup/spinquest/Anvesh/data/multi_track/processed_files

mkdir -p ${OUTDIR}

RUN="${APPTAINER} exec --bind ${CODEDIR}:/mnt/code --bind ${DONGHWA}:/mnt/donghwa --bind /project/ptgroup/spinquest/Anvesh:/mnt/anvesh ${IMAGE}"

# --- Generate signal training data ---
echo "Generate finder training data (train set)"
${RUN} python3 /mnt/code/data/multi_track/gen_training_random.py \
    /mnt/donghwa/data/raw_files/Dimuons_Train_500K_track1.root \
    /mnt/donghwa/data/raw_files/Dimuons_Train_500K_track2.root \
    --output /mnt/anvesh/data/multi_track/processed_files/finder_training_train.root \
    --pairsmup 5 \
    --pairsmum 5 \
    --random

echo "Generate finder training data (val set)"
${RUN} python3 /mnt/code/data/multi_track/gen_training_random.py \
    /mnt/donghwa/data/raw_files/Dimuons_Val_200K_track1.root \
    /mnt/donghwa/data/raw_files/Dimuons_Val_200K_track2.root \
    --output /mnt/anvesh/data/multi_track/processed_files/finder_training_val.root \
    --pairsmup 5 \
    --pairsmum 5 \
    --random

# --- Inject background tracks (low/med/high complexity) ---
echo "Inject low-level background (train)"
${RUN} python3 /mnt/code/data/multi_track/messy_gen.py \
    /mnt/anvesh/data/multi_track/processed_files/finder_training_train.root \
    /mnt/donghwa/data/processed_files/single_muons_train.root \
    --output /mnt/anvesh/data/multi_track/processed_files/mc_events_train_low.root \
    --uniform_tracks 0 \
    --lower_bound 0 \
    --num_tracks 16

echo "Inject mid-level background (train)"
${RUN} python3 /mnt/code/data/multi_track/messy_gen.py \
    /mnt/anvesh/data/multi_track/processed_files/finder_training_train.root \
    /mnt/donghwa/data/processed_files/single_muons_train.root \
    --output /mnt/anvesh/data/multi_track/processed_files/mc_events_train_med.root \
    --uniform_tracks 0 \
    --lower_bound 17 \
    --num_tracks 33

echo "Inject high-level background (train)"
${RUN} python3 /mnt/code/data/multi_track/messy_gen.py \
    /mnt/anvesh/data/multi_track/processed_files/finder_training_train.root \
    /mnt/donghwa/data/processed_files/single_muons_train.root \
    --output /mnt/anvesh/data/multi_track/processed_files/mc_events_train_high.root \
    --uniform_tracks 0 \
    --lower_bound 34 \
    --num_tracks 50

echo "Inject background (val)"
${RUN} python3 /mnt/code/data/multi_track/messy_gen.py \
    /mnt/anvesh/data/multi_track/processed_files/finder_training_val.root \
    /mnt/donghwa/data/processed_files/single_muons_val.root \
    --output /mnt/anvesh/data/multi_track/processed_files/mc_events_val.root

echo "Preprocessing complete. Output in ${OUTDIR}"
