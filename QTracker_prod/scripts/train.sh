# If using Rivanna, train using train.slurm instead

LEARNING_RATE=0.00005

python models/TrackFinder_unet.py \
 data/processed_files/mc_events_train.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_unet.h5 \
 --learning_rate $LEARNING_RATE \
 --batch_norm 1 \
 --dropout_bn 0.5 \
 --dropout_enc 0.4 \
 --backbone None

python evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_unet.h5

python evaluate.py \
 data/processed_files/mc_events_val_high.root \
 checkpoints/track_finder_unet.h5