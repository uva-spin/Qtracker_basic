import argparse
import numpy as np
import tensorflow as tf

from data_loader import load_data_denoise
from losses import custom_loss, weighted_bce

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def check_array(name, arr):
    arr = np.asarray(arr)
    print(f"\n[{name}]")
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    print("  has NaN:", np.isnan(arr).any())
    print("  has Inf:", np.isinf(arr).any())
    if arr.size > 0:
        print("  min:", np.nanmin(arr), "max:", np.nanmax(arr))


def main(args):
    print("\nLoading data...")
    X, X_clean, y_muPlus, y_muMinus = load_data_denoise(args.root_file)

    if X is None:
        print("Failed to load data")
        return

    y = np.stack([y_muPlus, y_muMinus], axis=1)

    # ---- Basic global checks ----
    check_array("X", X)
    check_array("X_clean", X_clean)
    check_array("y", y)

    # ---- Label validity ----
    print("\n[Label range checks]")
    for head, name in [(0, "mu+"), (1, "mu-")]:
        y_head = y[:, head, :]
        bad_low = np.where(y_head < 0)
        bad_high = np.where(y_head >= NUM_ELEMENT_IDS)

        print(f"{name}: min={y_head.min()}, max={y_head.max()}")

        if bad_low[0].size > 0:
            print(f"  ❌ {name}: negative labels found")
            print("    sample indices:", bad_low[0][:10])
            print("    detector indices:", bad_low[1][:10])

        if bad_high[0].size > 0:
            print(f"  ❌ {name}: labels >= {NUM_ELEMENT_IDS} found")
            print("    sample indices:", bad_high[0][:10])
            print("    detector indices:", bad_high[1][:10])

    # ---- Per-batch loss check ----
    print("\n[Per-batch loss sanity check]")
    bce_loss = weighted_bce(pos_weight=args.pos_weight)

    batch_size = args.batch_size
    num_batches = int(np.ceil(len(X) / batch_size))

    for b in range(num_batches):
        start = b * batch_size
        end = min(len(X), start + batch_size)

        Xb = tf.convert_to_tensor(X[start:end], tf.float32)
        Xcb = tf.convert_to_tensor(X_clean[start:end], tf.float32)
        yb = tf.convert_to_tensor(y[start:end], tf.int32)

        try:
            # fake predictions for loss shape validation
            denoise_pred = tf.clip_by_value(Xcb, 1e-6, 1.0 - 1e-6)

            seg_pred = tf.one_hot(
                yb,
                depth=NUM_ELEMENT_IDS,
                axis=-1,
                dtype=tf.float32,
            )

            loss_denoise = bce_loss(Xcb, denoise_pred)
            loss_segment = custom_loss(yb, seg_pred)

            if tf.math.is_nan(loss_denoise) or tf.math.is_nan(loss_segment):
                print(f"\n❌ NaN loss detected in batch {b}")
                print("  denoise loss:", loss_denoise.numpy())
                print("  segment loss:", loss_segment.numpy())
                print("  sample indices:", list(range(start, end)))
                break

        except Exception as e:
            print(f"\n❌ Exception in batch {b}")
            print(e)
            print("  sample indices:", list(range(start, end)))
            break
    else:
        print("\n✅ No NaNs detected in per-batch loss check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check dataset for NaNs and label errors")
    parser.add_argument("root_file", type=str, help="Path to ROOT file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
