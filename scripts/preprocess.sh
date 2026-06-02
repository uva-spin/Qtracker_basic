# --- 0. Create train/val/test sets ---
echo "Skim JPsi train set"
python Util/skim.py data/original_files/JPsi_Dump_1M.root \
    --output_file data/raw_files/JPsi_Dump_Train.root \
    --max_events 456000

echo "Skim MUP train set"
python Util/skim.py \
    /project/ptgroup/spinquest/RUS_MC/MUP/MUP_Dump_Sept_9_25_500M.root \
    --output_file data/raw_files/MUP_Dump_Train.root \
    --max_events 220000000

echo "Skim MUM train set"
python Util/skim.py \
    /project/ptgroup/spinquest/RUS_MC/MUM/MUM_Dump_Sept_15_25_500M.root \
    --output_file data/raw_files/MUM_Dump_Train.root \
    --max_events 220000000

echo "Skim JPsi val set"
python Util/skim.py data/original_files/JPsi_Dump_1M.root \
    --output_file data/raw_files/JPsi_Dump_Val.root \
    --max_events 57000 \
    --start 456000

echo "Skim MUP val set"
python Util/skim.py data/original_files/MUP_Dump_1M.root \
    --output_file data/raw_files/MUP_Dump_Val.root \
    --max_events 27500000 \
    --start 220000000

echo "Skim MUM val set"
python Util/skim.py data/original_files/MUM_Dump_1M.root \
    --output_file data/raw_files/MUM_Dump_Val.root \
    --max_events 27500000 \
    --start 220000000

echo "Skim JPsi test set"
python Util/skim.py data/original_files/JPsi_Dump_1M.root \
    --output_file data/raw_files/JPsi_Dump_Test.root \
    --max_events 57000 \
    --start 513000

echo "Skim MUP test set"
python Util/skim.py data/original_files/MUP_Dump_1M.root \
    --output_file data/raw_files/MUP_Dump_Test.root \
    --max_events 27500000 \
    --start 247500000

echo "Skim MUM test set"
python Util/skim.py data/original_files/MUM_Dump_1M.root \
    --output_file data/raw_files/MUM_Dump_Test.root \
    --max_events 27500000 \
    --start 247500000

# --- 1. Split signal ROOT file into μ⁺ and μ⁻ tracks ---
echo "Separate JPsi train set"
python data/separate.py data/raw_files/JPsi_Dump_Train.root

echo "Separate JPsi val set"
python data/separate.py data/raw_files/JPsi_Dump_Val.root

echo "Separate JPsi test set"
python data/separate.py data/raw_files/JPsi_Dump_Test.root

# --- 2. Merge two single-muon ROOT files ---
echo "Combine MUP and MUM train set"
python data/combine.py data/raw_files/MUP_Dump_Train.root data/raw_files/MUM_Dump_Train.root \
    --output data/processed_files/single_muons_train.root \
    --max_output_events 220000000

echo "Combine MUP and MUM val set"
python data/combine.py data/raw_files/MUP_Dump_Val.root data/raw_files/MUM_Dump_Val.root \
    --output data/processed_files/single_muons_val.root \
    --max_output_events 27500000

echo "Combine MUP and MUM test set"
python data/combine.py data/raw_files/MUP_Dump_Test.root data/raw_files/MUM_Dump_Test.root \
    --output data/processed_files/single_muons_test.root \
    --max_output_events 27500000

# --- 3. Generate training data by combining μ⁺ and μ⁻ signal tracks ---
echo "Generate training data for train set"
python data/gen_training.py \
    data/raw_files/JPsi_Dump_Train_track1.root \
    data/raw_files/JPsi_Dump_Train_track2.root \
    --output data/processed_files/finder_training_train.root

mv momentum_training-1.root data/processed_files/momentum_training-1_train.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_train.root

echo "Generate training data for val set"
python data/gen_training.py \
    data/raw_files/JPsi_Dump_Val_track1.root \
    data/raw_files/JPsi_Dump_Val_track2.root \
    --output data/processed_files/finder_training_val.root

mv momentum_training-1.root data/processed_files/momentum_training-1_val.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_val.root

echo "Generate training data for test set"
python data/gen_training.py \
    data/raw_files/JPsi_Dump_Test_track1.root \
    data/raw_files/JPsi_Dump_Test_track2.root \
    --output data/processed_files/finder_training_test.root

mv momentum_training-1.root data/processed_files/momentum_training-1_test.root 
mv momentum_training-2.root data/processed_files/momentum_training-2_test.root

# --- 4. Inject background muon tracks to signal file from 3 ---
echo "Inject low-level background tracks into train set"
python data/messy_gen.py \
    data/processed_files/finder_training_train.root \
    data/processed_files/single_muons_train.root \
    --output data/processed_files/mc_events_train_low.root \
    --uniform_tracks 0 \
    --lower_bound 0 \
    --num_tracks 16

echo "Inject mid-level background tracks into train set"
python data/messy_gen.py \
    data/processed_files/finder_training_train.root \
    data/processed_files/single_muons_train.root \
    --output data/processed_files/mc_events_train_med.root \
    --uniform_tracks 0 \
    --lower_bound 17 \
    --num_tracks 33

echo "Inject high-level background tracks into train set"
python data/messy_gen.py \
    data/processed_files/finder_training_train.root \
    data/processed_files/single_muons_train.root \
    --output data/processed_files/mc_events_train_high.root \
    --uniform_tracks 0 \
    --lower_bound 34 \
    --num_tracks 50

echo "Inject background tracks into val set"
python data/messy_gen.py \
    data/processed_files/finder_training_val.root \
    data/processed_files/single_muons_val.root \
    --output data/processed_files/mc_events_val.root
