# --- 0. Create train/val/test sets ---
python Util/skim.py data/raw_files/JPsi_Dump_100K.root \
 --output_file data/raw_files/JPsi_Dump_Train.root \
 --max_events 80

python Util/skim.py data/raw_files/MUP_Dump_100K.root \
 --output_file data/raw_files/MUP_Dump_Train.root \
 --max_events 80

python Util/skim.py data/raw_files/MUM_Dump_100K.root \
 --output_file data/raw_files/MUM_Dump_Train.root \
 --max_events 80

python Util/skim.py data/raw_files/JPsi_Dump_100K.root \
 --output_file data/raw_files/JPsi_Dump_Val.root \
 --max_events 10 \
 --start 100

python Util/skim.py data/raw_files/MUP_Dump_100K.root \
 --output_file data/raw_files/MUP_Dump_Val.root \
 --max_events 10 \
 --start 100

python Util/skim.py data/raw_files/MUM_Dump_100K.root \
 --output_file data/raw_files/MUM_Dump_Val.root \
 --max_events 10 \
 --start 100

python Util/skim.py data/raw_files/JPsi_Dump_100K.root \
 --output_file data/raw_files/JPsi_Dump_Test.root \
 --max_events 10 \
 --start 200

python Util/skim.py data/raw_files/MUP_Dump_100K.root \
 --output_file data/raw_files/MUP_Dump_Test.root \
 --max_events 10 \
 --start 200

python Util/skim.py data/raw_files/MUM_Dump_100K.root \
 --output_file data/raw_files/MUM_Dump_Test.root \
 --max_events 10 \
 --start 200

# --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
python data/separate.py data/raw_files/JPsi_Dump_Train.root
python data/separate.py data/raw_files/JPsi_Dump_Val.root
python data/separate.py data/raw_files/JPsi_Dump_Test.root


# --- 2. Merge two single-muon ROOT files ---
python data/combine.py data/raw_files/MUP_Dump_Train.root data/raw_files/MUM_Dump_Train.root \
 --output data/processed_files/single_muons_train.root \
 --max_output_events 80

python data/combine.py data/raw_files/MUP_Dump_Val.root data/raw_files/MUM_Dump_Val.root \
 --output data/processed_files/single_muons_val.root \
 --max_output_events 10

python data/combine.py data/raw_files/MUP_Dump_Test.root data/raw_files/MUM_Dump_Test.root \
 --output data/processed_files/single_muons_test.root \
 --max_output_events 10

# --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
python data/gen_training.py \
 data/raw_files/JPsi_Dump_Train_track1.root \
 data/raw_files/JPsi_Dump_Train_track2.root \
 --output data/processed_files/finder_training_train.root

mv momentum_training-1.root data/processed_files/momentum_training-1_train.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_train.root

python data/gen_training.py \
 data/raw_files/JPsi_Dump_Val_track1.root \
 data/raw_files/JPsi_Dump_Val_track2.root \
 --output data/processed_files/finder_training_val.root

mv momentum_training-1.root data/processed_files/momentum_training-1_val.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_val.root

python data/gen_training.py \
 data/raw_files/JPsi_Dump_Test_track1.root \
 data/raw_files/JPsi_Dump_Test_track2.root \
 --output data/processed_files/finder_training_test.root

mv momentum_training-1.root data/processed_files/momentum_training-1_test.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_test.root


# --- 4. Inject background muon tracks to signal file from 3 ---
python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_low.root \
 --uniform_tracks 0 \
 --num_tracks 10 \
 --lower_bound 0

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_med.root \
 --uniform_tracks 0 \
 --num_tracks 20 \
 --lower_bound 11

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_high.root \
 --uniform_tracks 0 \
 --num_tracks 30 \
 --lower_bound 21

python data/messy_gen.py \
 data/processed_files/finder_training_val.root \
 data/processed_files/single_muons_val.root \
 --output data/processed_files/mc_events_val.root