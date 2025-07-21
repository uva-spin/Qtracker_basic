LEARNING_RATE=0.00005

python training_scripts/TrackFinder_unet.py \
 data/processed_files/mc_events.root \
 --output_model models/track_finder_unet.h5 \
 --learning_rate $LEARNING_RATE \
 --batch_norm 1 \
 --dropout_bn 0.5 \
 --dropout_enc 0.4 \
 --backbone resnet50

# python evaluate.py \
#  data/processed_files/mc_events.root \
#  models/track_finder_unet.h5