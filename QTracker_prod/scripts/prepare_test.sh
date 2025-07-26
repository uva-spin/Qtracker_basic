# --- 0. Skim background tracks for training set ---
python Util/skim.py data/original_files/MUP_Dump_1M.root \
 --output_file data/raw_files/MUP_Dump_100K.root \
 --max_events 100000 \
 --from_last 1

python Util/skim.py data/original_files/MUM_Dump_1M.root \
 --output_file data/raw_files/MUM_Dump_100K.root \
 --max_events 100000 \
 --from_last 1

python Util/skim.py data/original_files/JPsi_Target_100K.root \
 --output_file data/raw_files/JPsi_Target_10K.root \
 --max_events 10000 \
 --from_last 1

# --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
python data/separate.py data/raw_files/JPsi_Target_10K.root


# --- 2. Merge two single-muon ROOT files ---
MAX_OUTPUT_EVENTS=100000

python data/combine.py data/raw_files/MUP_Dump_100K.root data/raw_files/MUM_Dump_100K.root \
 --output data/processed_files/single_muons_test.root \
 --max_output_events $MAX_OUTPUT_EVENTS


# --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
python data/gen_training.py \
 data/raw_files/JPsi_Target_10K_track1.root \
 data/raw_files/JPsi_Target_10K_track2.root \
 --output data/processed_files/finder_training_test.root

mv momentum_training-1.root data/processed_files/momentum_training-1.root
mv momentum_training-2.root data/processed_files/momentum_training-2.root


# --- 4. Inject background muon tracks to signal file from 3 ---
python data/messy_gen.py \
 data/processed_files/finder_training_test.root \
 data/processed_files/single_muons_test.root \
 --output data/processed_files/mc_events_test.root \
 --test_mode 1
