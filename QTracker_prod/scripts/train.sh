# If using Rivanna, train using train.slurm instead

LEARNING_RATE=0.0003
BATCH_NORM=1

# UNet-3+
python3 models/TrackFinder_unet_3p.py \
 data/processed_files/mc_events_train.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_unet_3p.h5 \
 --learning_rate $LEARNING_RATE \
 --patience 12 \
 --batch_norm $BATCH_NORM \
 --base 64

python3 evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_unet_3p.h5