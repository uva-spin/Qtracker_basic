# If using Rivanna, train using train.slurm instead

LEARNING_RATE=0.0003
BATCH_NORM=1

# UNet++ w/ deep supervision
python3 models/TrackFinder_unetpp.py \
 data/processed_files/mc_events_train.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_unetpp.h5 \
 --learning_rate $LEARNING_RATE \
 --patience 10 \
 --batch_norm $BATCH_NORM \
 --base 64 \
 --epochs 50 \
 --batch_size 32 \
 --dropout_bn 0.5 \
 --dropout_enc 0.4

python3 evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_unetpp.h5

# UNet 3+ w/ deep supervision
python3 models/TrackFinder_unet_3p.py \
 data/processed_files/mc_events_train.root \
 data/processed_files/mc_events_val.root \
 --output_model checkpoints/track_finder_unet_3p.h5 \
 --learning_rate $LEARNING_RATE \
 --patience 10 \
 --batch_norm $BATCH_NORM \
 --base 16 \
 --epochs 50 \
 --batch_size 32 \
 --dropout_bn 0.5 \
 --dropout 0.4 \
 --gradient_accumulation_steps 2

python3 evaluate.py \
 data/processed_files/mc_events_val.root \
 checkpoints/track_finder_unet_3p.h5