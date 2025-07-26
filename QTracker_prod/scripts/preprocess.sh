# --- 0. Skim background tracks for training set ---
python Util/skim.py data/original_files/MUP_Dump_1M.root \
 --output_file data/raw_files/MUP_Dump_900K.root \
 --max_events 900000

python Util/skim.py data/original_files/MUM_Dump_1M.root \
 --output_file data/raw_files/MUM_Dump_900K.root \
 --max_events 900000

# --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
python data/separate.py data/raw_files/JPsi_Target_100K.root


# --- 2. Merge two single-muon ROOT files ---
MAX_OUTPUT_EVENTS=900000

python data/combine.py data/raw_files/MUP_Dump_900K.root data/raw_files/MUM_Dump_900K.root \
 --output data/processed_files/single_muons.root \
 --max_output_events $MAX_OUTPUT_EVENTS


# --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
python data/gen_training.py \
 data/raw_files/JPsi_Target_100K_track1.root \
 data/raw_files/JPsi_Target_100K_track2.root \
 --output data/processed_files/finder_training.root

mv momentum_training-1.root data/processed_files/momentum_training-1.root
mv momentum_training-2.root data/processed_files/momentum_training-2.root


# --- 4. Inject background muon tracks to signal file from 3 ---
python data/messy_gen.py \
 data/processed_files/finder_training.root \
 data/processed_files/single_muons.root \
 --output data/processed_files/mc_events.root \

python Util/plot_HitMatrix.py data/processed_files/mc_events.root -event 42

# # --- 5. Inject randomly generated noise hits into file from 4 ---
# P_ELECTRONIC_NOISE=0.01
# P_CLUSTER_NOISE=0.05
# CLUSTER_LENGTH_RANGE="(2,4)"

# python data/noisy_gen.py data/mc_events.root \
#  --output data/noisy_output.root \
#  --p_electronic_noise $P_ELECTRONIC_NOISE \
#  --p_cluster_noise $P_CLUSTER_NOISE \
#  --cluster_length_range $CLUSTER_LENGTH_RANGE
