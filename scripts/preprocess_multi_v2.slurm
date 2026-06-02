#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J preprocess_1_pair
#SBATCH -o Slurm_Files/preprocess_1_pair.out
#SBATCH -e Slurm_Files/preprocess_1_pair.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

NUM_PAIRS=1
AT_LEAST_ONE_PAIR=0

# --- 1. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
echo "Generate training data for train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/gen_training_random.py \
        data/raw_files/Dimuons_Train_500K_track1.root \
        data/raw_files/Dimuons_Train_500K_track2.root \
        --output data/multi_track/processed_files/finder_training_train_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        --pairsmup $NUM_PAIRS \
        --pairsmum $NUM_PAIRS \
        --random

mv momentum_training-1.root data/multi_track/processed_files/momentum_training-1_train.root 
mv momentum_training-2.root data/multi_track/processed_files/momentum_training-2_train.root

echo "Generate training data for val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/gen_training_random.py \
        data/raw_files/Dimuons_Val_200K_track1.root \
        data/raw_files/Dimuons_Val_200K_track2.root \
        --output data/multi_track/processed_files/finder_training_val_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        --pairsmup $NUM_PAIRS \
        --pairsmum $NUM_PAIRS \
        --random

mv momentum_training-1.root data/multi_track/processed_files/momentum_training-1_val.root 
mv momentum_training-2.root data/multi_track/processed_files/momentum_training-2_val.root

# --- 2. Inject background muon tracks to signal file from 4 ---
echo "Inject low-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        data/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_low_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        --uniform_tracks 0 \
        --lower_bound 0 \
        --num_tracks 16

echo "Inject mid-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        data/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_med_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        --uniform_tracks 0 \
        --lower_bound 17 \
        --num_tracks 33
        
echo "Inject high-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        data/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_high_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        --uniform_tracks 0 \
        --lower_bound 34 \
        --num_tracks 50

echo "Inject background tracks into val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_val_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root \
        data/processed_files/single_muons_val.root \
        --output data/multi_track/processed_files/mc_events_val_${NUM_PAIRS}_${AT_LEAST_ONE_PAIR}.root
