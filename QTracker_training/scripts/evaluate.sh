# --- Plot Invariant Mass ---
python Util/imass_plot.py \
 data/processed_files/qtracker_reco.root \
 --output_plot plots/mass_plot.png

# --- Evaluate Momentum Residuals ---
python evaluate_momentum.py data/processed_files/qtracker_reco.root
 