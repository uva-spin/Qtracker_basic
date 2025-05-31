LEARNING_RATE=0.00005
EPOCH=40
BATCH_SIZE=32
PATIENCE=5


### Data Preprocessing ###
# Prerequisite: Create finder_training.root and single_muons.root using dimuon target and single muon dump
./preprocess.sh


### Model Training ###
python training_scripts/TrackFinder_prod.py data/noisy_output.root \
 --output_model models/track_finder.keras \
 --learning_rate $LEARNING_RATE \
 --epoch $EPOCH \
 --batch_size $BATCH_SIZE \
 --patience $PATIENCE

# python training_scripts/TrackFinder_acc.py data/noisy_output.root \
#  --output_model models/track_finder_resnet.keras \
#  --learning_rate $LEARNING_RATE \
#  --epoch $EPOCH \
#  --batch_size $BATCH_SIZE \
#  --patience $PATIENCE

# python training_scripts/TrackFinder_attention.py data/noisy_output.root \
#  --output_model models/track_finder_cbam.keras \
#  --learning_rate $LEARNING_RATE \
#  --epoch $EPOCH \
#  --batch_size $BATCH_SIZE \
#  --patience $PATIENCE


### MODEL EVALUATION ###
python evaluate.py data/noisy_output.root models/track_finder.keras
# python evaluate.py data/noisy_output.root models/track_finder_resnet.keras
# python evaluate.py data/noisy_output.root models/track_finder_cbam.keras
