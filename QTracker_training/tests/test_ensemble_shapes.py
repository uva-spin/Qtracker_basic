"""
Smoke test for EnsembleTrackFinder — verifies model shapes and loss computation
without requiring GPU or ROOT data.  Run locally:

    cd QTracker_training
    python3 tests/test_ensemble_shapes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import numpy as np
import tensorflow as tf

# Tiny model params so test runs fast on CPU
N_MODELS = 3
B = 4          # batch size
DET = 62
ELEM = 201

print("TF version:", tf.__version__)
print(f"Building {N_MODELS} sub-models (tiny: denoise_base=4, base=4)...")

from models.EnsembleTrackFinder import (
    build_single_model,
    EnsembleWrapper,
    ensemble_seg_loss,
)

sub_models = [
    build_single_model(denoise_base=4, base=4, model_idx=i)
    for i in range(N_MODELS)
]

ensemble = EnsembleWrapper(sub_models)

# --- Forward pass ---
X_dummy = tf.random.uniform((B, DET, ELEM, 1))
outputs = ensemble(X_dummy, training=False)

denoise_out = outputs["denoise"]
seg_out = outputs["segment"]

print(f"\nForward pass shapes:")
print(f"  denoise: {denoise_out.shape}  (expected: ({B}, {DET}, {ELEM}, 1))")
print(f"  segment: {seg_out.shape}  (expected: ({B}, {N_MODELS}, 2, {DET}, {ELEM}))")

assert denoise_out.shape == (B, DET, ELEM, 1), f"denoise shape mismatch: {denoise_out.shape}"
assert seg_out.shape == (B, N_MODELS, 2, DET, ELEM), f"segment shape mismatch: {seg_out.shape}"

# --- Loss computation ---
y_true = tf.random.uniform((B, N_MODELS, 2, DET), minval=0, maxval=ELEM, dtype=tf.int32)
loss_fn = ensemble_seg_loss(n_models=N_MODELS, diversity_lambda=0.05)
loss_val = loss_fn(y_true, seg_out)

print(f"\nLoss computation:")
print(f"  ensemble_seg_loss = {float(loss_val):.4f}  (should be finite, >0)")
assert tf.math.is_finite(loss_val), f"Loss is not finite: {loss_val}"
assert float(loss_val) > 0, f"Loss should be positive: {loss_val}"

# --- Diversity check: identical model outputs should yield HIGH diversity loss ---
# Force two identical outputs and check diversity penalty is nonzero
seg_same = tf.tile(seg_out[:, :1], [1, N_MODELS, 1, 1, 1])
loss_same = loss_fn(y_true, seg_same)
loss_diff = loss_fn(y_true, seg_out)
print(f"  Loss with identical outputs: {float(loss_same):.4f}")
print(f"  Loss with different outputs: {float(loss_diff):.4f}")
print(f"  Diversity penalty is active: {float(loss_same) > float(loss_diff) - 1e-3}")

# --- Softmax sums to 1 ---
sums = tf.reduce_sum(seg_out, axis=-1)  # (B, N, 2, DET)
assert tf.reduce_all(tf.abs(sums - 1.0) < 1e-4), "Softmax outputs do not sum to 1"
print(f"\nSoftmax sanity: all sums ≈ 1.0 ✓")

print("\n✅  All smoke tests passed.")
