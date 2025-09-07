# If using Rivanna, train using train.slurm instead

LEARNING_RATE=0.0003
BATCH_NORM=1

# UNet++
python3 models/TrackFinder_unetpp.py \
 data/processed_files/mc_events_train_low.root \
 data/processed_files/mc_events_train_med.root \
 data/processed_files/mc_events_train_high.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_unetpp.h5 \
 --learning_rate $LEARNING_RATE \
 --patience 10 \
 --batch_norm $BATCH_NORM \
 --base 64 \
 --epochs 4 \
 --batch_size 8 \
 --dropout_bn 0.5 \
 --dropout_enc 0.4

python3 evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_unetpp.h5