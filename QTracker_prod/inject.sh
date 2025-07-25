python data/inject.py \
 data/processed_files/finder_training.root \
 data/processed_files/single_muons.root \
 --output data/processed_files/mc_events.root

python Util/plot_HitMatrix.py data/processed_files/mc_events.root -event 0