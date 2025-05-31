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


# python Util/plot_HitMatrix.py data/finder_training.root -event $EVENT

python data/messy_gen.py data/finder_training.root data/single_muons_large.root \
 --num_tracks $NUM_TRACKS \
 --prob_mean $PROB_MEAN \
 --prob_width $PROB_WIDTH \
 --propagation_model $PROPAGATION_MODEL \
 --gaussian_sigma $GAUSSIAN_SIGMA \
 --exp_decay_const $EXP_DECAY_CONST

# python Util/plot_HitMatrix.py data/mc_events.root -event $EVENT

python data/noisy_gen.py data/mc_events.root \
 --p_electronic_noise $P_ELECTRONIC_NOISE \
 --p_cluster_noise $P_CLUSTER_NOISE \
 --cluster_length_range $CLUSTER_LENGTH_RANGE

# python Util/plot_HitMatrix.py data/noisy_output.root -event $EVENT