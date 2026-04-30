#!/bin/bash
#SBATCH --job-name=qtracker_viz
#SBATCH --output=viz_%j.out
#SBATCH --error=viz_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --account=ptgroup
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Load apptainer module
module load apptainer

# Paths
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=${SLURM_SUBMIT_DIR}

# Data and model paths
DATA_FILE=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/data/processed_files/mc_events_val.root
MODEL_PATH=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/checkpoints/track_finder_flagship.keras

# Event index to visualize (change this as needed)
EVENT_IDX=${EVENT_IDX:-0}

# Output format (mp4 or gif)
FORMAT=${FORMAT:-mp4}

# Visualization type (single or multi)
VIZ_TYPE=${VIZ_TYPE:-single}

echo "========================================="
echo "QTracker Visualization Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Event: $EVENT_IDX"
echo "Format: $FORMAT"
echo "Type: $VIZ_TYPE"
echo "========================================="

# Run visualization
if [ "$VIZ_TYPE" == "single" ]; then
    echo "Creating single-track visualization..."
    apptainer exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/Util/visualize_single_track.py \
        "${DATA_FILE}" \
        "${MODEL_PATH}" \
        --event ${EVENT_IDX} \
        --format ${FORMAT}
elif [ "$VIZ_TYPE" == "multi" ]; then
    echo "Creating multi-track visualization..."
    MAX_STEPS=${MAX_STEPS:-5}
    CONF_THRESH=${CONF_THRESH:-0.5}
    
    apptainer exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
        python3 /mnt/code/Util/visualize_multitrack.py \
        "${DATA_FILE}" \
        "${MODEL_PATH}" \
        --event ${EVENT_IDX} \
        --max_steps ${MAX_STEPS} \
        --confidence_threshold ${CONF_THRESH} \
        --format ${FORMAT}
else
    echo "Error: VIZ_TYPE must be 'single' or 'multi'"
    exit 1
fi

EXIT_CODE=$?

echo "========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Visualization completed successfully!"
    echo "Output saved to: plots/animations/"
    ls -lh plots/animations/*.${FORMAT} 2>/dev/null | tail -5
else
    echo "✗ Visualization failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
