#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p standard
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J preprocess_old
#SBATCH -o Slurm_Files/preprocess_old.out
#SBATCH -e Slurm_Files/preprocess_old.err
#SBATCH --mem=128000

module purge

# For apptainer exec
APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

# # --- 0. Create train/val/test sets ---
# echo "Split dimuon file into train, val, and test sets"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/split.py \
#         /project/ptgroup/spinquest/QTracker-Data/JPsi-Target_10M.rus.root \
#         --train_output data/old/raw_files/jpsi_train.root \
#         --val_output data/old/raw_files/jpsi_val.root \
#         --test_output data/old/raw_files/jpsi_test.root

# # --- 1. Skim train and val files to reduce sample size (for downstream ML) ---
# echo "Sample 20,000 events from raw val file (simple random)"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/skim.py \
#         data/old/raw_files/jpsi_val.root \
#         --output_file data/old/raw_files/jpsi_val_20k.root \
#         --max_events 20000 \
#         --random 1

# # --- 2. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
# echo "Separate dimuons val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/separate.py data/old/raw_files/jpsi_val_20k.root

# # --- 4. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
# echo "Generate training data for val set"
# ${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
#     python3 data/gen_training.py \
#         data/old/raw_files/jpsi_val_20k_track1.root \
#         data/old/raw_files/jpsi_val_20k_track2.root \
#         --output data/old/processed_files/finder_training_val.root

# mv momentum_training-1.root data/old/processed_files/momentum_training-1_val.root 
# mv momentum_training-2.root data/old/processed_files/momentum_training-2_val.root

# --- 5. Inject background muon tracks to signal file from 4 ---
echo "Inject background tracks into val set"
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 data/messy_gen.py \
        data/old/processed_files/finder_training_val.root \
        data/processed_files/single_muons_val.root \
        --output data/old/processed_files/mc_events_val.root

# Run QTracker
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/QTracker.py \
     /mnt/code/data/old/processed_files/mc_events_val.root \
     --output_file /mnt/code/data/old/processed_files/qtracker_reco_old.root

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/Util/imass_plot.py \
     /mnt/code/data/old/processed_files/qtracker_reco_old.root \
     --output_plot /mnt/code/plots/invariant_mass_old.png

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/QTracker.py \
     /project/ptgroup/spinquest/Donghwa/Qtracker_basic_old/QTracker_prod/data/processed_files/mc_events_val.root \
     --output_file /mnt/code/data/processed_files/qtracker_reco_jpsi.root

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/Util/imass_plot.py \
     /mnt/code/data/processed_files/qtracker_reco_jpsi.root \
     --output_plot /mnt/code/plots/invariant_mass_jpsi.png