#!/usr/bin/env bash
# train_all.sh

python3 training_scripts/Momentum_training.py \
    data/momentum_training-1.root \
    --output models/mom_mum.h5

python3 training_scripts/Momentum_training.py \
    data/momentum_training-2.root \
    --output models/mom_mup.h5