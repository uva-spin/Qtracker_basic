### Full RUN logic ###

# ---Input Config for Sanity Checks ---
export TRACK_SIZE=100
export MOM_SIZE=100
export QTRACK_SIZE=100

export TRACK_SKIM_INPUT="JPsi_Target_100K"
export MOM_SKIM_INPUT="JPsi_Target_100K"
export TRACK_SKIM_SIZE=$(( TRACK_SIZE * 1000 ))
export MOM_SKIM_SIZE=$(( MOM_SIZE * 1000 ))

export TRACKFINDER_INPUT="JPsi_Target_${TRACK_SIZE}K"
export MOMENTUM_INPUT="JPsi_Target_${MOM_SIZE}K"
export QTRACKER_INPUT="JPsi_Target_${QTRACK_SIZE}K"

TRAIN=1     # 0 = False (eval mode); 1 = True (train mode)

if (( TRAIN )); then
    # --- Data Preprocessing ---
    # ./preprocess.sh

    # --- Model Training ---
    sbatch run_tuner.slurm
else
    # Model Evaluation
    ./evaluate.sh
fi
