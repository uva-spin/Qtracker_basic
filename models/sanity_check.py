# ruff: noqa: E402

import numpy as np
import tensorflow as tf
from data_loader import load_data_denoise
from layers import AxialAttention

tf.keras.mixed_precision.set_global_policy("float32")

VAL_FILE = "data/multi_track/processed_files/mc_events_val.root"
MODEL_PATH = "checkpoints/multi_track_finder.keras"

BATCH_SIZE = 64


# ==========================================================
# DEBUG VERSION OF YOUR ORIGINAL WEIGHTED BCE
# ==========================================================
def weighted_bce_debug(pos_weight: float = 1.0):

    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # -------------------------
        # Input checks
        # -------------------------
        tf.debugging.assert_all_finite(
            y_true, "y_true contains NaN or Inf"
        )

        tf.debugging.assert_all_finite(
            y_pred, "y_pred contains NaN or Inf BEFORE BCE"
        )

        tf.debugging.assert_greater_equal(
            tf.reduce_min(y_true),
            0.0,
            message="y_true has values < 0",
        )

        tf.debugging.assert_less_equal(
            tf.reduce_max(y_true),
            1.0,
            message="y_true has values > 1",
        )

        # Logit magnitude check
        max_logit = tf.reduce_max(tf.abs(y_pred))
        tf.debugging.assert_all_finite(
            max_logit, "logit magnitude invalid"
        )

        # -------------------------
        # Weight computation
        # -------------------------
        weights = 1.0 + (pos_weight - 1.0) * y_true

        tf.debugging.assert_all_finite(
            weights, "weights contain NaN or Inf"
        )

        # -------------------------
        # BCE computation
        # -------------------------
        bce_loss = bce(y_true, y_pred, sample_weight=weights)

        tf.debugging.assert_all_finite(
            bce_loss, "BCE output contains NaN or Inf"
        )

        final = tf.reduce_mean(bce_loss)

        tf.debugging.assert_all_finite(
            final, "Final reduced loss is NaN or Inf"
        )

        return final

    return loss


# ==========================================================
# MAIN NAN SCANNER
# ==========================================================
def main():

    print("Loading validation data...")
    X_val, X_clean_val, _, _ = load_data_denoise(
        VAL_FILE,
        multi_track=True,
        max_pairs=5,
    )

    if X_val is None:
        print("Validation data loading failed.")
        return

    print("Loading trained model...")
    custom_objects = {"AxialAttention": AxialAttention}
    model = tf.keras.models.load_model(
        "checkpoints/multi_track_finder.keras",
        compile=False,
        custom_objects=custom_objects,
    )

    loss_fn = weighted_bce_debug(pos_weight=20.0)

    n_samples = X_val.shape[0]
    n_batches = int(np.ceil(n_samples / BATCH_SIZE))

    print(f"\nScanning {n_samples} samples in {n_batches} batches\n")

    for i in range(n_batches):

        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, n_samples)

        x_batch = X_val[start:end]
        y_batch = X_clean_val[start:end]

        print(f"Batch {i+1}/{n_batches}")

        # Forward pass
        denoise_pred, _ = model(x_batch, training=False)

        # Extra raw output check
        if np.isnan(denoise_pred).any():
            print("\nNaN detected in raw model output")
            print("Min:", np.nanmin(denoise_pred))
            print("Max:", np.nanmax(denoise_pred))
            return

        # Compute debug loss
        loss = loss_fn(
            tf.convert_to_tensor(y_batch),
            tf.convert_to_tensor(denoise_pred),
        )

        print("  Loss:", loss.numpy())

    print("\nNo NaNs detected in entire validation set.")


if __name__ == "__main__":
    main()
