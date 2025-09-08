# --- 4. Inject background muon tracks to signal file from 3 ---
python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_clean.root \
 --num_tracks 10

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_medium.root \
 --num_tracks 20 \
 --lower_bound 10

python data/messy_gen.py \
 data/processed_files/finder_training_train.root \
 data/processed_files/single_muons_train.root \
 --output data/processed_files/mc_events_train_noisy.root \
 --num_tracks 30 \
 --lower_bound 20

python data/messy_gen.py \
 data/processed_files/finder_training_val.root \
 data/processed_files/single_muons_val.root \
 --output data/processed_files/mc_events_val_clean.root \
 --num_tracks 10

python data/messy_gen.py \
 data/processed_files/finder_training_val.root \
 data/processed_files/single_muons_val.root \
 --output data/processed_files/mc_events_val_medium.root \
 --num_tracks 20 \
 --lower_bound 10

python data/messy_gen.py \
 data/processed_files/finder_training_val.root \
 data/processed_files/single_muons_val.root \
 --output data/processed_files/mc_events_val_noisy.root \
 --num_tracks 30 \
 --lower_bound 20
