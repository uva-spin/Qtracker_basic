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
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_prod/

# # --- 0. Create train/val/test sets ---
# echo "Skim JPsi train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py data/original_files/JPsi_Dump_1M.root \
#         --output_file data/raw_files/JPsi_Dump_Train.root \
#         --max_events 456000

# echo "Skim MUP train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUP/MUP_Dump_Sept_9_25_500M.root \
#         --output_file data/raw_files/MUP_Dump_Train.root \
#         --max_events 220000000

# echo "Skim MUM train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUM/MUM_Dump_Sept_15_25_500M.root \
#         --output_file data/raw_files/MUM_Dump_Train.root \
#         --max_events 220000000

# echo "Skim JPsi val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py data/original_files/JPsi_Dump_1M.root \
#         --output_file data/raw_files/JPsi_Dump_Val.root \
#         --max_events 57000 \
#         --start 456000

# echo "Skim MUP val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUP/MUP_Dump_Sept_9_25_500M.root \
#         --output_file data/raw_files/MUP_Dump_Val.root \
#         --max_events 27500000 \
#         --start 220000000

# echo "Skim MUM val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUM/MUM_Dump_Sept_15_25_500M.root \
#         --output_file data/raw_files/MUM_Dump_Val.root \
#         --max_events 27500000 \
#         --start 220000000

# echo "Skim JPsi test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py data/original_files/JPsi_Dump_1M.root \
#         --output_file data/raw_files/JPsi_Dump_Test.root \
#         --max_events 57000 \
#         --start 513000

# echo "Skim MUP test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUP/MUP_Dump_Sept_9_25_500M.root \
#         --output_file data/raw_files/MUP_Dump_Test.root \
#         --max_events 27500000 \
#         --start 247500000

# echo "Skim MUM test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 Util/skim_v2.py \
#         /project/ptgroup/spinquest/RUS_MC/MUM/MUM_Dump_Sept_15_25_500M.root \
#         --output_file data/raw_files/MUM_Dump_Test.root \
#         --max_events 27500000 \
#         --start 247500000

# # --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
# echo "Separate JPsi train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/JPsi_Dump_Train.root

# echo "Separate JPsi val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/JPsi_Dump_Val.root

# echo "Separate JPsi test set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/raw_files/JPsi_Dump_Test.root

# # --- 2. Merge two single-muon ROOT files ---
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

# # --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
# echo "Generate training data for train set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" python3 data/gen_training.py data/raw_files/JPsi_Dump_Train_track1.root data/raw_files/JPsi_Dump_Train_track2.root --output data/processed_files/finder_training_train.root --pairs 2

# mv momentum_training-1.root data/processed_files/momentum_training-1_train.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_train.root

# echo "Generate training data for val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" python3 data/gen_training.py data/raw_files/JPsi_Dump_Val_track1.root data/raw_files/JPsi_Dump_Val_track2.root --output data/processed_files/finder_training_val.root --pairs 2

# mv momentum_training-1.root data/processed_files/momentum_training-1_val.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_val.root

# echo "Generate training data for test set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" python3 data/gen_training.py data/raw_files/JPsi_Dump_Test_track1.root data/raw_files/JPsi_Dump_Test_track2.root --output data/processed_files/finder_training_test.root --pairs 2

# mv momentum_training-1.root data/processed_files/momentum_training-1_test.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_test.root

# --- 4. Inject background muon tracks to signal file from 3 ---
# echo "Inject background tracks into train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#    python3 data/messy_gen_v4.py \
#        data/processed_files/finder_training_train.root \
#        data/processed_files/single_muons_train.root \
#        --output data/processed_files/mc_events_train.root \
#        --uniform_tracks 0 \

# echo "Inject low-level background tracks into train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/messy_gen_v4.py \
#         data/processed_files/finder_training_train.root \
#         data/processed_files/single_muons_train.root \
#         --output data/processed_files/mc_events_train_low.root \
#         --uniform_tracks 0 \
#         --lower_bound 0 \
#         --num_tracks 16

# echo "Inject mid-level background tracks into train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/messy_gen_v4.py \
#         data/processed_files/finder_training_train.root \
#         data/processed_files/single_muons_train.root \
#         --output data/processed_files/mc_events_train_med.root \
#         --uniform_tracks 0 \
#         --lower_bound 17 \
#         --num_tracks 33
        
# echo "Inject high-level background tracks into train set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/messy_gen_v4.py \
#         data/processed_files/finder_training_train.root \
#         data/processed_files/single_muons_train.root \
#         --output data/processed_files/mc_events_train_high.root \
#         --uniform_tracks 0 \
#         --lower_bound 34 \
#         --num_tracks 50

# echo "Inject background tracks into val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/messy_gen_v4.py \
#         data/processed_files/finder_training_val.root \
#         data/processed_files/single_muons_val.root \
#         --output data/processed_files/mc_events_val.root