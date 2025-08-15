# If using Rivanna, train using train.slurm instead

LEARNING_RATE=0.00005
BATCH_NORM=1

python3 models/TrackFinder_UGNN_v4.py \
 data/processed_files/mc_events_train.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_UGNN_v4.h5 \

python3 evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_UGNN_v4.h5

python3 evaluate.py \
 data/processed_files/mc_events_val_high.root \
 checkpoints/track_finder_UGNN_v4.h5 
