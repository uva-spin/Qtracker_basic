#!/bin/bash
#SBATCH -A spinquest_standard
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -J ensemble-fno-denoiser
#SBATCH -o Slurm_Files/ensemble-fno-denoiser.out
#SBATCH -e Slurm_Files/ensemble-fno-denoiser.err
#SBATCH --mem=256000

module purge

APPTAINER=/apps/software/standard/core/apptainer/1.4.5/bin/apptainer
IMAGE=/project/ptgroup/spinquest/David/TfRootBuild_v2.sif
_DEFAULT_CODEDIR=/scratch/am4qw/Qtracker_basic/QTracker_training
CODEDIR=${CODEDIR:-${_DEFAULT_CODEDIR}}
DATADIR=${DATADIR:-/project/ptgroup/spinquest/Anvesh}

### Main Body ###
# Short diagnostic run, not the full curriculum: ensemble_fno_v1 (base=32,
# fno_base=32, full 100-epoch schedule) overfit hard in the low phase by
# epoch 28 (train segment_nonempty_acc climbing, val flat, val_loss
# increasing) -- same signature as the widened pair-count classifier, and it
# then hit the 48h walltime before reaching med/high anyway. Testing reduced
# per-sub-model capacity (base/fno_base 32->16) cheaply first: 20 epochs,
# mostly low phase, before committing to another multi-day run.
RUN_TAG=ensemble_fno_v2_reduced_capacity
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=/mnt/data/checkpoints/ensemble_fno_denoiser/${RUN_NAME}

${APPTAINER} exec --nv --pwd /mnt/code --env TF_FORCE_GPU_ALLOW_GROWTH=true --env MLFLOW_TRACKING_URI=file:///mnt/data/mlruns --bind ${CODEDIR}:/mnt/code --bind ${DATADIR}:/mnt/data "${IMAGE}" \
    python3 /mnt/code/models/experiments/fno_denoiser/ensemble_fno_denoiser.py \
     /mnt/data/data/multi_track/processed_files/mc_events_train_low.root \
     /mnt/data/data/multi_track/processed_files/mc_events_val.root \
     --train_root_file_med /mnt/data/data/multi_track/processed_files/mc_events_train_med.root \
     --train_root_file_high /mnt/data/data/multi_track/processed_files/mc_events_train_high.root \
     --output_dir ${OUTPUT_DIR} \
     --n_models 3 \
     --diversity_lambda 0.05 \
     --batch_norm 0 \
     --use_attn 1 \
     --use_attn_ffn 0 \
     --dropout_bn 0.5 \
     --dropout_enc 0.2 \
     --dropout_attn 0.1 \
     --base 16 \
     --fno_base 16 \
     --fno_depth 4 \
     --k_max 16 \
     --num_heads 4 \
     --pos_weight 20.0 \
     --epochs 20 \
     --low_ratio 0.75 \
     --med_ratio 0.9 \
     --batch_size 8 \
     --patience 20 \
     --mlflow_experiment ensemble_fno_denoiser \
     --mlflow_run_name "${RUN_NAME}"
