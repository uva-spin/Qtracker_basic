#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J preprocess
#SBATCH -o Slurm_Files/preprocess.out
#SBATCH -e Slurm_Files/preprocess.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

# --- 0. Create train/val/test sets ---
echo "Split dimuon file into train, val, and test sets"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/split.py \
        /project/ptgroup/spinquest/QTracker-Data/Target_acc_Pythia8.rus.root \
        --train_output data/raw_files/Dimuons_Train.root \
        --val_output data/raw_files/Dimuons_Val.root \
        --test_output data/raw_files/Dimuons_Test.root

# echo "Split muon positive file into train, val, and test sets"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/split.py \
#         /project/ptgroup/spinquest/QTracker-Data/MUP-Dump_500M.rus.root \
#         --train_output data/raw_files/MUP_Dump_Train.root \
#         --val_output data/raw_files/MUP_Dump_Val.root \
#         --test_output data/raw_files/MUP_Dump_Test.root

# echo "Split muon negative file into train, val, and test sets"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/split.py \
#         /project/ptgroup/spinquest/QTracker-Data/MUM-Dump_500M.rus.root \
#         --train_output data/raw_files/MUM_Dump_Train.root \
#         --val_output data/raw_files/MUM_Dump_Val.root \
#         --test_output data/raw_files/MUM_Dump_Test.root

# --- 1. Skim train and val files to reduce sample size (for downstream ML) ---
echo "Sample 500,000 events from raw train file (stratified)"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/skim_flat.py \
        data/raw_files/Dimuons_Train.root \
        data/raw_files/Dimuons_Train_500K.root

echo "Sample 200,000 events from raw val file (simple random)"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/skim.py \
        data/raw_files/Dimuons_Val.root \
        --output_file data/raw_files/Dimuons_Val_200K.root \
        --max_events 200000 \
        --random 1

# --- 2. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
echo "Separate dimuons train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/separate.py data/raw_files/Dimuons_Train_500K.root

echo "Separate dimuons val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/separate.py data/raw_files/Dimuons_Val_200K.root

echo "Separate dimuons test set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/separate.py data/raw_files/Dimuons_Test.root

# # --- 3. Merge two single-muon ROOT files ---
# echo "Combine MUP and MUM train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/combine.py data/raw_files/MUP_Dump_Train.root data/raw_files/MUM_Dump_Train.root \
#         --output data/processed_files/single_muons_train.root \
#         --max_output_events 220000000

# echo "Combine MUP and MUM val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/combine.py data/raw_files/MUP_Dump_Val.root data/raw_files/MUM_Dump_Val.root \
#         --output data/processed_files/single_muons_val.root \
#         --max_output_events 27500000

# echo "Combine MUP and MUM test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/combine.py data/raw_files/MUP_Dump_Test.root data/raw_files/MUM_Dump_Test.root \
#         --output data/processed_files/single_muons_test.root \
#         --max_output_events 27500000

# --- 4. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
echo "Generate training data for train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/gen_training.py \
        data/raw_files/Dimuons_Train_500K_track1.root \
        data/raw_files/Dimuons_Train_500K_track2.root \
        --output data/processed_files/finder_training_train.root

mv momentum_training-1.root data/processed_files/momentum_training-1_train.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_train.root

echo "Generate training data for val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/gen_training.py \
        data/raw_files/Dimuons_Val_200K_track1.root \
        data/raw_files/Dimuons_Val_200K_track2.root \
        --output data/processed_files/finder_training_val.root

mv momentum_training-1.root data/processed_files/momentum_training-1_val.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_val.root

echo "Generate training data for test set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/gen_training.py \
        data/raw_files/Dimuons_Test_track1.root \
        data/raw_files/Dimuons_Test_track2.root \
        --output data/processed_files/finder_training_test.root

mv momentum_training-1.root data/processed_files/momentum_training-1_test.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_test.root

# --- 5. Inject background muon tracks to signal file from 4 ---
echo "Inject low-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/messy_gen.py \
        data/processed_files/finder_training_train.root \
        data/processed_files/single_muons_train.root \
        --output data/processed_files/mc_events_train_low.root \
        --uniform_tracks 0 \
        --lower_bound 0 \
        --num_tracks 16

echo "Inject mid-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/messy_gen.py \
        data/processed_files/finder_training_train.root \
        data/processed_files/single_muons_train.root \
        --output data/processed_files/mc_events_train_med.root \
        --uniform_tracks 0 \
        --lower_bound 17 \
        --num_tracks 33
        
echo "Inject high-level background tracks into train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/messy_gen.py \
        data/processed_files/finder_training_train.root \
        data/processed_files/single_muons_train.root \
        --output data/processed_files/mc_events_train_high.root \
        --uniform_tracks 0 \
        --lower_bound 34 \
        --num_tracks 50

echo "Inject background tracks into val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/messy_gen.py \
        data/processed_files/finder_training_val.root \
        data/processed_files/single_muons_val.root \
        --output data/processed_files/mc_events_val.root
