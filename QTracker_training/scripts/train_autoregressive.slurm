#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpupod
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J autoregressive-finetune
#SBATCH -o Slurm_Files/autoregressive-finetune.out
#SBATCH -e Slurm_Files/autoregressive-finetune.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.3.4/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild.sif
CODEDIR=/project/ptgroup/spinquest/Donghwa/Qtracker_basic/QTracker_training/

BATCH_NORM=1
USE_ATTENTION=1
USE_ATTENTION_FFN=0

### Step 1: Generate residual training data from multi-track files ###
echo "Generating residual training data (train)..."
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/multi_track/gen_residual_training.py \
     /mnt/code/data/multi_track/processed_files/mc_events_train_low.root \
     --output /mnt/code/data/multi_track/processed_files/residual_train_low.root \
     --max_pairs 5

echo "Generating residual training data (val)..."
${APPTAINER} exec --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/data/multi_track/gen_residual_training.py \
     /mnt/code/data/multi_track/processed_files/mc_events_val.root \
     --output /mnt/code/data/multi_track/processed_files/residual_val.root \
     --max_pairs 5

### Step 2: Fine-tune single-track model on residual data ###
# Uses the standard single-track TrackFinder training script
# since residual data is in single-track format
${APPTAINER} exec --nv --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/TrackFinder.py \
     /mnt/code/data/multi_track/processed_files/residual_train_low.root \
     /mnt/code/data/multi_track/processed_files/residual_val.root \
     --output_model /mnt/code/checkpoints/track_finder_autoregressive.keras \
     --lr_low 0.0001 \
     --patience 10 \
     --lr_patience 3 \
     --batch_norm $BATCH_NORM \
     --use_attn $USE_ATTENTION \
     --use_attn_ffn $USE_ATTENTION_FFN \
     --denoise_base 32 \
     --base 64 \
     --epochs 30 \
     --batch_size 64 \
     --dropout_bn 0.5 \
     --dropout_enc 0.4 \
     --dropout_attn 0.1 \
     --factor 0.3 \
     --pos_weight 20.0

### Step 3: Run autoregressive eval with fine-tuned model ###
${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/models/AutoregressiveTrackFinder.py \
     /mnt/code/data/multi_track/processed_files/mc_events_val.root \
     /mnt/code/checkpoints/track_finder_autoregressive.keras \
     --max_iters 5 \
     --confidence_threshold 0.15 \
     --max_pairs 5 \
     --output /mnt/code/checkpoints/autoregressive_finetuned_predictions.npz

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 /mnt/code/eval_autoregressive.py \
     /mnt/code/checkpoints/autoregressive_finetuned_predictions.npz
