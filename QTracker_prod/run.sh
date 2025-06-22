### Full RUN logic ###

# ---Input Config for Sanity Checks ---
export TRACKFINDER_INPUT="JPsi_Dump_10K"
export MOMENTUM_INPUT="JPsi_Dump_10K"
export QTRACKER_INPUT="JPsi_Dump_10K"

# --- Data Preprocessing ---
./preprocess.sh

# --- Model Training and Evaluation ---
sbatch run_tuner.slurm
