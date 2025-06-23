# --- Evaluate Track Finder ---
python evaluate.py \
 data/processed_files/finder_training.root \
 models/track_finder.h5

# --- Training Q-tracker ---
# Uncomment if running run_tuner.slurm directly
# QTRACK_SIZE=20
# QTRACKER_INPUT="JPsi_Dump_${QTRACK_SIZE}K"

python QTracker_prod.py \
 data/raw_files/${QTRACKER_INPUT}.root \
 --output_file data/processed_files/qtracker_reco.root

# --- Plot Invariant Mass ---
python Util/imass_plot.py \
 data/processed_files/qtracker_reco.root \
 --output_plot plots/mass_plot_${TRACK_SIZE}_${MOM_SIZE}_${QTRACK_SIZE}.png

# --- Evaluate Momentum Residuals ---
python evaluate_mom.py data/processed_files/qtracker_reco.root

# Clean-up Work
mv models/track_finder.h5 models/track_finder_${TRACK_SIZE}T_${MOM_SIZE}T_${QTRACK_SIZE}T.h5
mv models/mom_mup.h5 models/mom_mup_${TRACK_SIZE}T_${MOM_SIZE}T_${QTRACK_SIZE}T.h5
mv models/mom_mum.h5 models/mom_mum_${TRACK_SIZE}T_${MOM_SIZE}T_${QTRACK_SIZE}T.h5
 