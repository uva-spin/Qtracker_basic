# --- Evaluate Track Finder ---
python3 evaluate.py \
 data/processed_files/finder_training.root \
 models/track_finder.h5

# --- Training Q-tracker ---
# Uncomment if running run_tuner.slurm directly
# QTRACK_SIZE=20
# QTRACKER_INPUT="JPsi_Dump_${QTRACK_SIZE}K"

python3 /mnt/code/QTracker_prod.py \
 /mnt/code/data/raw_files/${QTRACKER_INPUT}.root \
 --output_file /mnt/code/data/processed_files/qtracker_reco.root

# --- Plot Invariant Mass ---
python3 /mnt/code/Util/imass_plot.py \
 data/processed_files/qtracker_reco.root \
 --output_plot plots/mass_plot_${TRACK_SIZE}_${MOM_SIZE}_${QTRACK_SIZE}.png
 