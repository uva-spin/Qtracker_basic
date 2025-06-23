# Variables: Uncomment if running preprocess.sh directly
# TRACK_SIZE=20
# MOM_SIZE=10

# TRACK_SKIM_INPUT="JPsi_Dump_100K"
# MOM_SKIM_INPUT="JPsi_Dump_100K"
# TRACK_SKIM_SIZE=$(( TRACK_SIZE * 1000 ))
# MOM_SKIM_SIZE=$(( MOM_SIZE * 1000 ))

# TRACKFINDER_INPUT="JPsi_Dump_${TRACK_SIZE}K"
# MOMENTUM_INPUT="JPsi_Dump_${MOM_SIZE}K"


# --- 0. Skim input ROOT files if necessary ---
python Util/skim.py data/original_files/${TRACK_SKIM_INPUT}.root \
 --output data/raw_files/${TRACKFINDER_INPUT}.root \
 --max_events ${TRACK_SKIM_SIZE}

python Util/skim.py data/original_files/${MOM_SKIM_INPUT}.root \
 --output data/raw_files/${MOMENTUM_INPUT}.root \
 --max_events ${MOM_SKIM_SIZE}


# --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
python data/separate.py data/raw_files/${TRACKFINDER_INPUT}.root
python data/separate.py data/raw_files/${MOMENTUM_INPUT}.root


# # --- 2. Merge two single-muon ROOT files ---
# MAX_OUTPUT_EVENTS=100000

# python data/combine.py data/raw_files/MUP_Dump_100K.root data/raw_files/MUM_Dump_100K.root \
#  --output data/processed_files/single_muons.root \
#  --max_output_events $MAX_OUTPUT_EVENTS


# --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
python data/gen_training.py \
 data/raw_files/${TRACKFINDER_INPUT}_track1.root \
 data/raw_files/${TRACKFINDER_INPUT}_track2.root \
 --output data/processed_files/finder_training.root

rm momentum_training-1.root
rm momentum_training-2.root

python data/gen_training.py \
 data/raw_files/${MOMENTUM_INPUT}_track1.root \
 data/raw_files/${MOMENTUM_INPUT}_track2.root \
 --output data/processed_files/finder_training_unused.root

rm data/processed_files/finder_training_unused.root
rm data/processed_files/momentum_training-1.root
rm data/processed_files/momentum_training-2.root
mv momentum_training-1.root data/processed_files/
mv momentum_training-2.root data/processed_files/


# # --- 4. Inject background muon tracks to signal file from 3 ---
# NUM_TRACKS=5
# CLEAN_RATIO=0.2
# PROB_MEAN=0.9
# PROB_WIDTH=0.1
# PROPAGATION_MODEL="gaussian"
# GAUSSIAN_SIGMA=10.0
# EXP_DECAY_CONST=15.0

# python data/messy_gen.py \
#  data/processed_files/finder_training.root \
#  data/processed_files/single_muons_small.root \
#  --output data/processed_files/mc_events.root \
#  --num_tracks $NUM_TRACKS \
#  --clean_ratio $CLEAN_RATIO \
#  --prob_mean $PROB_MEAN \
#  --prob_width $PROB_WIDTH \
#  --propagation_model $PROPAGATION_MODEL \
#  --gaussian_sigma $GAUSSIAN_SIGMA \
#  --exp_decay_const $EXP_DECAY_CONST


# # --- 5. Inject randomly generated noise hits into file from 4 ---
# P_ELECTRONIC_NOISE=0.01
# P_CLUSTER_NOISE=0.05
# CLUSTER_LENGTH_RANGE="(2,4)"

# python data/noisy_gen.py data/mc_events.root \
#  --output data/noisy_output.root \
#  --p_electronic_noise $P_ELECTRONIC_NOISE \
#  --p_cluster_noise $P_CLUSTER_NOISE \
#  --cluster_length_range $CLUSTER_LENGTH_RANGE
