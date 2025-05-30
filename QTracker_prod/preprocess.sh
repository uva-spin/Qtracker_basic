python Util/plot_HitMatrix.py data/finder_training.root -event $EVENT

python data/messy_gen.py data/finder_training.root data/single_muons.root \
 --num_tracks $NUM_TRACKS \
 --prob_mean $PROB_MEAN \
 --prob_width $PROB_WIDTH \
 --propagation_model $PROPAGATION_MODEL \
 --gaussian_sigma $GAUSSIAN_SIGMA \
 --exp_decay_const $EXP_DECAY_CONST

python Util/plot_HitMatrix.py data/mc_events.root -event $EVENT

python data/noisy_gen.py data/mc_events.root \
 --p_electronic_noise $P_ELECTRONIC_NOISE \
 --p_cluster_noise $P_CLUSTER_NOISE \
 --cluster_length_range $CLUSTER_LENGTH_RANGE

python Util/plot_HitMatrix.py data/noisy_output.root -event $EVENT