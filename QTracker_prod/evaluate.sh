# --- Training Q-tracker ---
python QTracker_prod.py \
 data/original_files/JPsi_Target_100K.root \
 --output_file data/processed_files/qtracker_reco.root

# --- Plot Invariant Mass ---
python Util/imass_plot.py \
 data/processed_files/qtracker_reco.root \
 --output_plot plots/mass_plot.png

# --- Evaluate Momentum Residuals ---
python evaluate_mom.py data/processed_files/qtracker_reco.root
 