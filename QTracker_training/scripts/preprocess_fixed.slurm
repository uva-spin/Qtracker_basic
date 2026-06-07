#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J preprocess_fixed
#SBATCH -o Slurm_Files/preprocess_fixed.out
#SBATCH -e Slurm_Files/preprocess_fixed.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

# # --- 0. Create train/val/test sets ---
# echo "Split dimuon file into train, val, and test sets"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/split.py \
#         /project/ptgroup/spinquest/QTracker-Data/Dimuons-Target_19M.rus.root \
#         --train_output data/raw_files/Dimuons_Train.root \
#         --val_output data/raw_files/Dimuons_Val.root \
#         --test_output data/raw_files/Dimuons_Test.root

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

# # --- 1. Skim train and val files to reduce sample size (for downstream ML) ---
# echo "Sample 500,000 events from raw train file (stratified)"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/skim_flat.py \
#         data/raw_files/Dimuons_Train.root \
#         data/raw_files/Dimuons_Train_500K.root

# echo "Sample 200,000 events from raw val file (simple random)"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/skim.py \
#         data/raw_files/Dimuons_Val.root \
#         --output_file data/raw_files/Dimuons_Val_200K.root \
#         --max_events 200000 \
#         --random 1

# # --- 2. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
# echo "Separate dimuons train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/Dimuons_Train_500K.root

# echo "Separate dimuons val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/Dimuons_Val_200K.root

# echo "Separate dimuons test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/Dimuons_Test.root

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

# --- 4a. Generate random multi-track data for the counter ---
echo "Generate random training data for train set (counter)"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/gen_training_random.py \
        data/raw_files/Dimuons_Train_500K_track1.root \
        data/raw_files/Dimuons_Train_500K_track2.root \
        --output data/multi_track/processed_files/finder_training_train_random.root \
        --pairsmup 5 \
        --pairsmum 5 \
        --random

mv momentum_training-1.root data/multi_track/processed_files/momentum_training-1_train.root
mv momentum_training-2.root data/multi_track/processed_files/momentum_training-2_train.root

echo "Generate random training data for val set (counter)"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/gen_training_random.py \
        data/raw_files/Dimuons_Val_200K_track1.root \
        data/raw_files/Dimuons_Val_200K_track2.root \
        --output data/multi_track/processed_files/finder_training_val_random.root \
        --pairsmup 5 \
        --pairsmum 5 \
        --random

mv momentum_training-1.root data/multi_track/processed_files/momentum_training-1_val.root
mv momentum_training-2.root data/multi_track/processed_files/momentum_training-2_val.root

# --- 4b. Generate fixed-pair training data for N=1, 2, 3 ---
for N in 1 2 3; do
    echo "Generate fixed-N training data (N=${N}) for train set"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/data/multi_track/gen_fixed_pairs.py \
            data/raw_files/Dimuons_Train_500K_track1.root \
            data/raw_files/Dimuons_Train_500K_track2.root \
            --output data/multi_track/processed_files/finder_training_train_N${N}.root \
            --n_pairs ${N} \
            --pairsmup 5 \
            --pairsmum 5
    # discard momentum files for finder training (not needed here)
    rm -f momentum_training-1.root momentum_training-2.root

    echo "Generate fixed-N training data (N=${N}) for val set"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/data/multi_track/gen_fixed_pairs.py \
            data/raw_files/Dimuons_Val_200K_track1.root \
            data/raw_files/Dimuons_Val_200K_track2.root \
            --output data/multi_track/processed_files/finder_training_val_N${N}.root \
            --n_pairs ${N} \
            --pairsmup 5 \
            --pairsmum 5
    rm -f momentum_training-1.root momentum_training-2.root
done

# --- 5a. Inject background into random (counter) training data ---
echo "Inject low-level background tracks into random train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_random.root \
        data/multi_track/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_low_random.root \
        --uniform_tracks 0 \
        --lower_bound 0 \
        --num_tracks 16

echo "Inject mid-level background tracks into random train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_random.root \
        data/multi_track/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_med_random.root \
        --uniform_tracks 0 \
        --lower_bound 17 \
        --num_tracks 33

echo "Inject high-level background tracks into random train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_train_random.root \
        data/multi_track/processed_files/single_muons_train.root \
        --output data/multi_track/processed_files/mc_events_train_high_random.root \
        --uniform_tracks 0 \
        --lower_bound 34 \
        --num_tracks 50

echo "Inject background tracks into random val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/multi_track/messy_gen.py \
        data/multi_track/processed_files/finder_training_val_random.root \
        data/multi_track/processed_files/single_muons_val.root \
        --output data/multi_track/processed_files/mc_events_val_random.root

# --- 5b. Inject background into fixed-pair training data for N=1, 2, 3 ---
for N in 1 2 3; do
    echo "Inject low-level background tracks into fixed-N train set (N=${N})"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 data/multi_track/messy_gen.py \
            data/multi_track/processed_files/finder_training_train_N${N}.root \
            data/multi_track/processed_files/single_muons_train.root \
            --output data/multi_track/processed_files/mc_events_train_low_N${N}.root \
            --uniform_tracks 0 \
            --lower_bound 0 \
            --num_tracks 16

    echo "Inject mid-level background tracks into fixed-N train set (N=${N})"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 data/multi_track/messy_gen.py \
            data/multi_track/processed_files/finder_training_train_N${N}.root \
            data/multi_track/processed_files/single_muons_train.root \
            --output data/multi_track/processed_files/mc_events_train_med_N${N}.root \
            --uniform_tracks 0 \
            --lower_bound 17 \
            --num_tracks 33

    echo "Inject high-level background tracks into fixed-N train set (N=${N})"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 data/multi_track/messy_gen.py \
            data/multi_track/processed_files/finder_training_train_N${N}.root \
            data/multi_track/processed_files/single_muons_train.root \
            --output data/multi_track/processed_files/mc_events_train_high_N${N}.root \
            --uniform_tracks 0 \
            --lower_bound 34 \
            --num_tracks 50

    echo "Inject background tracks into fixed-N val set (N=${N})"
    ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 data/multi_track/messy_gen.py \
            data/multi_track/processed_files/finder_training_val_N${N}.root \
            data/multi_track/processed_files/single_muons_val.root \
            --output data/multi_track/processed_files/mc_events_val_N${N}.root
done
