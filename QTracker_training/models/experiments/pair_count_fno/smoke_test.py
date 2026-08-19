"""
Local smoke test for the Axial-FNO pair-count classifier.

No ROOT file and no GPU required -- uses random tensors to check that the
model builds, the forward pass produces the right shape, and gradients
flow through every trainable weight (including the FNO spectral weights).
Run this before spending Rivanna allocation on a real training job.

Runs under both float32 and mixed_float16 policies -- train.py always sets
mixed_float16, and a prior bug (autocast=True default on add_weight() would
silently downcast the FNO spectral weights to float16 inside call(), which
tf.complex() rejects) only manifested under that policy, so testing float32
alone isn't representative of what actually runs on Rivanna.

Usage:
    python3 smoke_test.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

MAX_PAIRS = 3
BATCH = 4


def run(policy: str) -> None:
    print(f"\n{'=' * 60}\nPolicy: {policy}\n{'=' * 60}")
    mixed_precision.set_global_policy(policy)

    # Re-import after the policy is set, since layer dtype policy is
    # captured at construction time.
    import importlib
    import model as model_module
    importlib.reload(model_module)
    from model import NUM_DETECTORS, NUM_ELEMENT_IDS

    m = model_module.build_model(max_pairs=MAX_PAIRS, base=16, fno_depth=2, k_max=16, num_heads=2)

    rng = np.random.default_rng(0)
    X = (rng.random((BATCH, NUM_DETECTORS, NUM_ELEMENT_IDS, 1)) < 0.05).astype(np.float32)
    y = rng.integers(0, MAX_PAIRS + 1, size=(BATCH,)).astype(np.int32)

    with tf.GradientTape() as tape:
        preds = m(X, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
        loss = tf.reduce_mean(loss)

    assert preds.shape == (BATCH, MAX_PAIRS + 1), f"unexpected output shape {preds.shape}"
    print(f"Forward pass OK -- output shape {preds.shape}, dtype {preds.dtype}, loss {float(loss):.4f}")

    grads = tape.gradient(loss, m.trainable_variables)
    n_missing = sum(g is None for g in grads)
    n_nonfinite = sum(
        1 for g in grads if g is not None and not tf.reduce_all(tf.math.is_finite(g))
    )
    assert n_missing == 0, f"{n_missing} trainable variables received no gradient"
    assert n_nonfinite == 0, f"{n_nonfinite} gradients contain non-finite values"
    print(f"Gradient check OK -- {len(grads)} trainable variables, all finite gradients")

    spectral_vars = [v for v in m.trainable_variables if "spectral_weight" in v.name]
    assert len(spectral_vars) == 2 * 2, "expected w_real + w_imag per FourierBlock1D"
    assert all(v.dtype == tf.float32 for v in spectral_vars), (
        f"spectral weights must stay float32 under {policy}, got "
        f"{[(v.name, v.dtype) for v in spectral_vars if v.dtype != tf.float32]}"
    )
    print(f"Found {len(spectral_vars)} FNO spectral weight tensors, all float32 as required")


def main() -> None:
    run("float32")
    run("mixed_float16")
    print("\nSmoke test passed under both policies.")


if __name__ == "__main__":
    main()
