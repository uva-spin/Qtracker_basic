"""
Local smoke test for the Axial-FNO pair-count classifier.

No ROOT file and no GPU required -- uses random tensors to check that the
model builds, the forward pass produces the right shape, and gradients
flow through every trainable weight (including the FNO spectral weights).
Run this before spending Rivanna allocation on a real training job.

Usage:
    python3 smoke_test.py
"""

import numpy as np
import tensorflow as tf

from model import build_model, NUM_DETECTORS, NUM_ELEMENT_IDS

MAX_PAIRS = 3
BATCH = 4


def main() -> None:
    model = build_model(max_pairs=MAX_PAIRS, base=16, fno_depth=2, k_max=16, num_heads=2)
    model.summary()

    rng = np.random.default_rng(0)
    X = (rng.random((BATCH, NUM_DETECTORS, NUM_ELEMENT_IDS, 1)) < 0.05).astype(np.float32)
    y = rng.integers(0, MAX_PAIRS + 1, size=(BATCH,)).astype(np.int32)

    with tf.GradientTape() as tape:
        preds = model(X, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        loss = tf.reduce_mean(loss)

    assert preds.shape == (BATCH, MAX_PAIRS + 1), f"unexpected output shape {preds.shape}"
    print(f"\nForward pass OK -- output shape {preds.shape}, loss {float(loss):.4f}")

    grads = tape.gradient(loss, model.trainable_variables)
    n_missing = sum(g is None for g in grads)
    n_nonfinite = sum(
        1 for g in grads if g is not None and not tf.reduce_all(tf.math.is_finite(g))
    )
    assert n_missing == 0, f"{n_missing} trainable variables received no gradient"
    assert n_nonfinite == 0, f"{n_nonfinite} gradients contain non-finite values"
    print(f"Gradient check OK -- {len(grads)} trainable variables, all finite gradients")

    spectral_vars = [v for v in model.trainable_variables if "spectral_weight" in v.name]
    print(f"Found {len(spectral_vars)} FNO spectral weight tensors (expected {2 * 2})")
    assert len(spectral_vars) == 2 * 2, "expected w_real + w_imag per FourierBlock1D"

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
