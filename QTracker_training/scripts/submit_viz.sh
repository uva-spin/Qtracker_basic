#!/bin/bash
# Helper script to submit visualization jobs with common parameters

# Usage examples:
# ./submit_viz.sh                           # Single-track, event 0, MP4
# ./submit_viz.sh --event 5 --type multi    # Multi-track, event 5, MP4
# ./submit_viz.sh --event 3 --format gif    # Single-track, event 3, GIF

# Default values
EVENT_IDX=0
VIZ_TYPE="single"
FORMAT="mp4"
MAX_STEPS=5
CONF_THRESH=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --event)
            EVENT_IDX="$2"
            shift 2
            ;;
        --type)
            VIZ_TYPE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --confidence-threshold)
            CONF_THRESH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--event N] [--type single|multi] [--format mp4|gif] [--max-steps N] [--confidence-threshold X]"
            exit 1
            ;;
    esac
done

# Navigate to QTracker_training directory
cd "$(dirname "$0")/.." || exit 1

echo "Submitting visualization job..."
echo "  Event: $EVENT_IDX"
echo "  Type: $VIZ_TYPE"
echo "  Format: $FORMAT"
if [ "$VIZ_TYPE" == "multi" ]; then
    echo "  Max steps: $MAX_STEPS"
    echo "  Confidence threshold: $CONF_THRESH"
fi

# Submit job with environment variables
JOB_ID=$(sbatch \
    --export=ALL,EVENT_IDX=$EVENT_IDX,VIZ_TYPE=$VIZ_TYPE,FORMAT=$FORMAT,MAX_STEPS=$MAX_STEPS,CONF_THRESH=$CONF_THRESH \
    scripts/visualize.slurm | awk '{print $4}')

echo ""
echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f viz_${JOB_ID}.out"
echo ""
echo "Output will be in: plots/animations/"
