"""
Local smoke test for the FNO denoiser. No ROOT file and no GPU required.

Runs under both float32 and mixed_float16 -- train.py always sets
mixed_float16, and the FNO spectral weights only break under that policy
(see pair_count_fno's smoke_test.py / EXPERIMENTS.md for the autocast=True
bug this already caused once).

Usage:
    python3 smoke_test.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

BATCH = 4


def run(policy: str) -> None:
    print(f"\n{'=' * 60}\nPolicy: {policy}\n{'=' * 60}")
    mixed_precision.set_global_policy(policy)

    import importlib
    import model as model_module
    importlib.reload(model_module)
    from model import NUM_DETECTORS, NUM_ELEMENT_IDS

    m = model_module.build_denoiser(base=16, fno_depth=2, k_max=16, num_heads=2)

    rng = np.random.default_rng(0)
    X = (rng.random((BATCH, NUM_DETECTORS, NUM_ELEMENT_IDS, 1)) < 0.1).astype(np.float32)
    X_clean = (rng.random((BATCH, NUM_DETECTORS, NUM_ELEMENT_IDS, 1)) < 0.02).astype(np.float32)

    with tf.GradientTape() as tape:
        logits = m(X, training=True)
        loss = tf.keras.losses.binary_crossentropy(X_clean, logits, from_logits=True)
        loss = tf.reduce_mean(loss)

    assert logits.shape == (BATCH, NUM_DETECTORS, NUM_ELEMENT_IDS, 1), f"unexpected output shape {logits.shape}"
    print(f"Forward pass OK -- output shape {logits.shape}, dtype {logits.dtype}, loss {float(loss):.4f}")

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
