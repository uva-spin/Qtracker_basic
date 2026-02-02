import ROOT  # noqa: F401
import numpy as np
import tensorflow as tf

from backbones import unetpp_backbone
from data_loader import load_data_denoise
from losses import custom_loss, weighted_bce

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201
BATCH = 4  # small on purpose


def build_model():
    from tensorflow.keras import layers

    inp = layers.Input(shape=(NUM_DETECTORS, NUM_ELEMENT_IDS, 1))

    x = unetpp_backbone(
        inp,
        NUM_DETECTORS,
        NUM_ELEMENT_IDS,
        use_bn=False,
        dropout_bn=0.0,
        dropout_enc=0.0,
        base=64,
        use_attn=False,
    )

    denoise = layers.Conv2D(1, 1, activation="sigmoid", name="denoise")(x)

    x = unetpp_backbone(
        denoise,
        NUM_DETECTORS,
        NUM_ELEMENT_IDS,
        use_bn=False,
        dropout_bn=0.0,
        dropout_enc=0.0,
        base=64,
        use_attn=False,
    )

    x = layers.Conv2D(2, 1)(x)
    x = layers.Softmax(axis=2)(x)
    segment = layers.Permute((3, 1, 2), name="segment")(x)

    return tf.keras.Model(inp, [denoise, segment])


def check(name, arr):
    print(
        f"{name:20s}",
        arr.shape,
        arr.dtype,
        "min",
        np.nanmin(arr),
        "max",
        np.nanmax(arr),
        "nan",
        np.isnan(arr).any(),
        "inf",
        np.isinf(arr).any(),
    )


def main(train_root):
    # Load data
    print("=== LOADING DATA ===")
    X, X_clean, y_muP, y_muM = load_data_denoise(train_root)

    assert X is not None

    y = np.stack([y_muP, y_muM], axis=1)

    # Slice tiny batch
    X = X[:BATCH]
    X_clean = X_clean[:BATCH]
    y = y[:BATCH]

    # Sanity checks
    print("\n=== DATA CHECK ===")
    check("X", X)
    check("X_clean", X_clean)
    check("y_muPlus", y_muP[:BATCH])
    check("y_muMinus", y_muM[:BATCH])

    # Build model
    model = build_model()

    # Forward pass
    print("\n=== FORWARD PASS ===")
    den_pred, seg_pred = model(X, training=False)

    tf.debugging.check_numerics(den_pred, "denoise output NaN")
    tf.debugging.check_numerics(seg_pred, "segment output NaN")

    print("denoise output:", den_pred.shape, den_pred.dtype)
    print("segment output:", seg_pred.shape, seg_pred.dtype)

    # Compute losses
    print("\n=== LOSS CHECK ===")
    den_loss_fn = weighted_bce(pos_weight=1.0)
    seg_loss_fn = custom_loss

    den_loss = den_loss_fn(X_clean, den_pred)
    seg_loss = seg_loss_fn(y, seg_pred)

    tf.debugging.check_numerics(den_loss, "denoise loss NaN")
    tf.debugging.check_numerics(seg_loss, "segment loss NaN")

    print("denoise loss:", float(den_loss))
    print("segment loss:", float(seg_loss))

    print("\nOK: no NaNs detected")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python nan_smoke_test.py <train_root_file>")
        sys.exit(1)

    main(sys.argv[1])
