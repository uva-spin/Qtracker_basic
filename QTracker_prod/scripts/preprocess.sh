# # --- 0. Create train/val/test sets ---
# python Util/skim.py data/original_files/JPsi_Dump_1M.root \
#  --output_file data/raw_files/JPsi_Dump_Train.root \
#  --max_events 370000

# python Util/skim.py data/original_files/MUP_Dump_1M.root \
#  --output_file data/raw_files/MUP_Dump_Train.root \
#  --max_events 800000

# python Util/skim.py data/original_files/MUM_Dump_1M.root \
#  --output_file data/raw_files/MUM_Dump_Train.root \
#  --max_events 800000

# python Util/skim.py data/original_files/JPsi_Dump_1M.root \
#  --output_file data/raw_files/JPsi_Dump_Val.root \
#  --max_events 100000 \
#  --start 370000

# python Util/skim.py data/original_files/MUP_Dump_1M.root \
#  --output_file data/raw_files/MUP_Dump_Val.root \
#  --max_events 100000 \
#  --start 800000

# python Util/skim.py data/original_files/MUM_Dump_1M.root \
#  --output_file data/raw_files/MUM_Dump_Val.root \
#  --max_events 100000 \
#  --start 800000

# python Util/skim.py data/original_files/JPsi_Dump_1M.root \
#  --output_file data/raw_files/JPsi_Dump_Test.root \
#  --max_events 100000 \
#  --start 470000

# python Util/skim.py data/original_files/MUP_Dump_1M.root \
#  --output_file data/raw_files/MUP_Dump_Test.root \
#  --max_events 100000 \
#  --start 900000

# python Util/skim.py data/original_files/MUM_Dump_1M.root \
#  --output_file data/raw_files/MUM_Dump_Test.root \
#  --max_events 100000 \
#  --start 900000

# # --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
# python data/separate.py data/raw_files/JPsi_Dump_Train.root
# python data/separate.py data/raw_files/JPsi_Dump_Val.root
# python data/separate.py data/raw_files/JPsi_Dump_Test.root


# # --- 2. Merge two single-muon ROOT files ---
# python data/combine.py data/raw_files/MUP_Dump_Train.root data/raw_files/MUM_Dump_Train.root \
#  --output data/processed_files/single_muons_train.root \
#  --max_output_events 800000

# python data/combine.py data/raw_files/MUP_Dump_Val.root data/raw_files/MUM_Dump_Val.root \
#  --output data/processed_files/single_muons_val.root \
#  --max_output_events 100000

# python data/combine.py data/raw_files/MUP_Dump_Test.root data/raw_files/MUM_Dump_Test.root \
#  --output data/processed_files/single_muons_test.root \
#  --max_output_events 100000

# # --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
# python data/gen_training.py \
#  data/raw_files/JPsi_Dump_Train_track1.root \
#  data/raw_files/JPsi_Dump_Train_track2.root \
#  --output data/processed_files/finder_training_train.root

# mv momentum_training-1.root data/processed_files/momentum_training-1_train.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_train.root

# python data/gen_training.py \
#  data/raw_files/JPsi_Dump_Val_track1.root \
#  data/raw_files/JPsi_Dump_Val_track2.root \
#  --output data/processed_files/finder_training_val.root

# mv momentum_training-1.root data/processed_files/momentum_training-1_val.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_val.root

# python data/gen_training.py \
#  data/raw_files/JPsi_Dump_Test_track1.root \
#  data/raw_files/JPsi_Dump_Test_track2.root \
#  --output data/processed_files/finder_training_test.root

# mv momentum_training-1.root data/processed_files/momentum_training-1_test.root 
# mv momentum_training-2.root data/processed_files/momentum_training-2_test.root


# --- 4. Inject background muon tracks to signal file from 3 ---
python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_low.root \
 --uniform_tracks 1 \
 --num_tracks 10 \
 --lower_bound 0

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_med.root \
 --uniform_tracks 1 \
 --num_tracks 20 \
 --lower_bound 11

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_high.root \
 --uniform_tracks 1 \
 --num_tracks 30 \
 --lower_bound 21

python data/messy_gen.py \
 data/processed_files/finder_training_val.root \
 data/processed_files/single_muons_val.root \
 --output data/processed_files/mc_events_val.root

# # --- 5. Inject randomly generated noise hits into file from 4 ---
# P_ELECTRONIC_NOISE=0.01
# P_CLUSTER_NOISE=0.05
# CLUSTER_LENGTH_RANGE="(2,4)"

# python data/noisy_gen.py data/mc_events.root \
#  --output data/noisy_output.root \
#  --p_electronic_noise $P_ELECTRONIC_NOISE \
#  --p_cluster_noise $P_CLUSTER_NOISE \
#  --cluster_length_range $CLUSTER_LENGTH_RANGE
