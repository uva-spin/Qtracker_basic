### Full RUN logic ###

# --- Data Preprocessing ---
./preprocess.sh

# --- Model Training and Evaluation ---
sbatch run_tuner.slurm
