# DO NOT ALTER UNLESS INSTRUCTED
PROB_MEAN=0.9
PROB_WIDTH=0.1
PROPAGATION_MODEL="gaussian"
GAUSSIAN_SIGMA=10.0
EXP_DECAY_CONST=15.0

P_ELECTRONIC_NOISE=0.01
P_CLUSTER_NOISE=0.05
CLUSTER_LENGTH_RANGE="(2,4)"

# Can modify
NUM_TRACKS=5
EVENT=42
LEARNING_RATE=0.00005
EPOCH=40
BATCH_SIZE=32
PATIENCE=5


### Data Preprocessing ###
# Prerequisite: Create finder_training.root and single_muons.root using dimuon target and single muon dump
./preprocess.sh


### Model Training ###
python training_scripts/TrackFinder_prod.py data/noisy_output.root \
 --learning_rate $LEARNING_RATE \
 --epoch $EPOCH \
 --batch_size $BATCH_SIZE \
 --patience $PATIENCE

python training_scripts/TrackFinder_acc.py data/noisy_output.root \
 --learning_rate $LEARNING_RATE \
 --epoch $EPOCH \
 --batch_size $BATCH_SIZE \
 --patience $PATIENCE

python training_scripts/TrackFinder_attention.py data/noisy_output.root \
 --learning_rate $LEARNING_RATE \
 --epoch $EPOCH \
 --batch_size $BATCH_SIZE \
 --patience $PATIENCE


### MODEL EVALUATION ###
python evaluate.py data/noisy_output.root models/track_finder.h5
python evaluate.py data/noisy_output.root models/track_finder_resnet.h5
python evaluate.py data/noisy_output.root models/track_finder_cbam.h5
