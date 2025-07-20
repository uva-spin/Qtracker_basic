### Full RUN logic ###
TRAIN=1     # 0 = False (eval mode); 1 = True (train mode)

if (( TRAIN )); then
    # --- Data Preprocessing ---
    ./preprocess.sh

    # --- Model Training ---
    sbatch run_tuner.slurm
else
    # Model Evaluation
    ./evaluate.sh
fi
