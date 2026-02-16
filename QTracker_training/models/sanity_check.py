import argparse
import numpy as np
import tensorflow as tf

from backbones import unetpp_backbone
from data_loader import load_data_denoise
from TrackFinder import build_model  # adjust if build_model is elsewhere

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def find_bad_training_batch(model, X, X_clean, y, bs, pos_weight=1.0):
    from losses import custom_loss, weighted_bce

    denoise_loss_fn = weighted_bce(pos_weight=pos_weight)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0003,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        xclean_b = X_clean[i:i + bs]
        yb = y[i:i + bs]

        with tf.GradientTape() as tape:
            den, seg = model(xb, training=True)

            loss_den = denoise_loss_fn(xclean_b, den)
            loss_seg = custom_loss(yb, seg)

            total_loss = 10.0 * loss_den + loss_seg

        grads = tape.gradient(total_loss, model.trainable_variables)

        # --- Check forward outputs ---
        if not tf.reduce_all(tf.math.is_finite(den)):
            print(f"\n❌ Non-finite DENOISE output in batch {i}")
            return i

        if not tf.reduce_all(tf.math.is_finite(seg)):
            print(f"\n❌ Non-finite SEGMENT output in batch {i}")
            return i

        # --- Check losses ---
        if not tf.math.is_finite(loss_den):
            print(f"\n❌ Non-finite denoise loss in batch {i}")
            return i

        if not tf.math.is_finite(loss_seg):
            print(f"\n❌ Non-finite segment loss in batch {i}")
            return i

        if not tf.math.is_finite(total_loss):
            print(f"\n❌ Non-finite TOTAL loss in batch {i}")
            return i

        # --- Check gradients ---
        for g in grads:
            if g is not None and not tf.reduce_all(tf.math.is_finite(g)):
                print(f"\n❌ Non-finite gradient in batch {i}")
                return i

        # Optional: simulate optimizer step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("✅ All batches stable under forward + loss + backprop")
    return None


def check_labels(y, name, num_classes=201):
    y = np.asarray(y)
    print(f"\n{name}")
    print("  dtype:", y.dtype)
    print("  min:", y.min(), "max:", y.max())

    bad = np.where((y < 0) | (y >= num_classes))
    print("  bad count:", bad[0].size)


def check_binary_targets(x, name):
    x = np.asarray(x)
    print(f"\n{name}")
    print("  dtype:", x.dtype)
    print("  min:", x.min(), "max:", x.max())

    non_binary = np.sum((x != 0) & (x != 1))
    print("  non-binary count:", non_binary)


def main(args):
    print("Loading training data...")
    X_train, X_clean_train, y_muPlus_train, y_muMinus_train = \
        load_data_denoise(args.train_root_file)

    print("Loading validation data...")
    X_val, X_clean_val, y_muPlus_val, y_muMinus_val = \
        load_data_denoise(args.val_root_file)

    # Label checks
    check_labels(y_muPlus_train, "y_muPlus_train")
    check_labels(y_muMinus_train, "y_muMinus_train")
    check_labels(y_muPlus_val, "y_muPlus_val")
    check_labels(y_muMinus_val, "y_muMinus_val")

    # Binary denoise target checks
    check_binary_targets(X_clean_train, "X_clean_train")
    check_binary_targets(X_clean_val, "X_clean_val")

    print("\nBuilding model...")
    model = build_model(
        num_detectors=NUM_DETECTORS,
        num_elementIDs=NUM_ELEMENT_IDS,
    )

    y_train = np.stack([y_muPlus_train, y_muMinus_train], axis=1)

    print("\nRunning forward-pass batch scan...")
    find_bad_training_batch(model, X_train, X_clean_train, y_train, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root_file", type=str)
    parser.add_argument("val_root_file", type=str)
    parser.add_argument("--batch_size", type=int, default=48)

    args = parser.parse_args()
    main(args)
