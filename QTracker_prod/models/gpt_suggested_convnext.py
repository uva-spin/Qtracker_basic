import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.applications import convnext as KNX

# ---------- tiny helpers ----------
def dec_block(x, skip, f, name):
    x = L.Conv2DTranspose(f, 3, strides=2, padding="same", name=f"{name}.up")(x)
    if skip is not None:
        x = L.Concatenate(name=f"{name}.cat")([x, skip])
    x = L.Conv2D(f, 3, padding="same", name=f"{name}.c1")(x)
    x = L.LayerNormalization(epsilon=1e-6, name=f"{name}.ln1")(x)
    x = L.Activation("gelu", name=f"{name}.a1")(x)
    x = L.Conv2D(f, 3, padding="same", name=f"{name}.c2")(x)
    x = L.LayerNormalization(epsilon=1e-6, name=f"{name}.ln2")(x)
    x = L.Activation("gelu", name=f"{name}.a2")(x)
    return x

def pick_four_stages(backbone):
    """Return 4 feature maps at ~/4, /8, /16, /32 (shallow->deep)."""
    feats, seen = [], set()
    for ly in backbone.layers:
        shp = getattr(ly, "output_shape", None)
        if isinstance(shp, tuple) and len(shp) == 4:
            h, w = shp[1], shp[2]
            if h and w and (h, w) not in seen:
                seen.add((h, w)); feats.append(ly.output)
    # last four distinct resolutions (ConvNeXt produces increasing downsampling)
    return feats[-4:]

# ---------- main builder ----------
def build_convnext_unet(
    input_shape=(512, 512, 3),
    num_classes=3,
    variant="tiny",            # "tiny" | "small" | "base" | "large"
    pretrained=False           # True -> ImageNet (requires 3 channels)
):
    assert input_shape[0] % 32 == 0 and input_shape[1] % 32 == 0, \
        "H and W should be divisible by 32 for ConvNeXt encoders."

    inp = L.Input(shape=input_shape)
    x = L.Rescaling(1./255.0)(inp)  # keep it simple; ConvNeXt is fine with [0..1]

    BB = {
        "tiny":  KNX.ConvNeXtTiny,
        "small": KNX.ConvNeXtSmall,
        "base":  KNX.ConvNeXtBase,
        "large": KNX.ConvNeXtLarge,
    }[variant.lower()]

    bb = BB(
        include_top=False,
        weights=("imagenet" if (pretrained and input_shape[-1] == 3) else None),
        input_tensor=x
    )

    C1, C2, C3, C4 = pick_four_stages(bb)  # shallow -> deep
    # bottleneck head
    b = L.Conv2D(512, 1, padding="same", name="bottleneck.c")(C4)
    b = L.LayerNormalization(epsilon=1e-6, name="bottleneck.ln")(b)
    b = L.Activation("gelu", name="bottleneck.a")(b)

    # decoder (three skips from C3, C2, C1)
    d3 = dec_block(b,  C3, 256, "dec3")   # up to ~ /16
    d2 = dec_block(d3, C2, 128, "dec2")   # up to ~ /8
    d1 = dec_block(d2, C1,  64, "dec1")   # up to ~ /4

    # back to native resolution (two more ups)
    d0 = L.Conv2DTranspose(32, 3, strides=2, padding="same", name="dec0.up")(d1)  # /2
    d0 = L.Conv2D(32, 3, padding="same", name="dec0.c")(d0)
    d0 = L.LayerNormalization(epsilon=1e-6, name="dec0.ln")(d0)
    d0 = L.Activation("gelu", name="dec0.a")(d0)

    out = L.Conv2DTranspose(32, 3, strides=2, padding="same", name="out.up")(d0)  # /1
    logits = L.Conv2D(num_classes, 1, padding="same", name="head")(out)
    probs = L.Softmax(name="softmax")(logits) if num_classes > 1 else L.Activation("sigmoid", name="sigmoid")(logits)

    return keras.Model(inp, probs, name=f"ConvNeXt-{variant}-UNet")

# ---------- compile (Dice + CE) ----------
def dice_loss(y_true, y_pred, eps=1e-6):
    C = tf.shape(y_pred)[-1]
    y_true_1h = tf.one_hot(tf.cast(y_true, tf.int32), C)
    inter = tf.reduce_sum(y_true_1h * y_pred, axis=[1,2])
    denom = tf.reduce_sum(y_true_1h + y_pred, axis=[1,2])
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - tf.reduce_mean(dice)

def make_model(input_shape=(512,512,3), num_classes=3, variant="tiny", pretrained=False):
    model = build_convnext_unet(input_shape, num_classes, variant, pretrained)
    ce = (keras.losses.SparseCategoricalCrossentropy() if num_classes > 1
          else keras.losses.BinaryCrossentropy())
    def loss_fn(y_true, y_pred): return 0.5 * ce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)
    model.compile(optimizer=keras.optimizers.AdamW(3e-4), loss=loss_fn, metrics=["accuracy"])
    return model

# Example:
# model = make_model(input_shape=(512,512,1), num_classes=3, variant="tiny", pretrained=False)
# model.summary()
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8)
