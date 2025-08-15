import os, sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

try:
    from .data_loader import load_data
    from .losses import custom_loss
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR  = os.path.dirname(CURRENT_DIR)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    from models.data_loader import load_data
    from models.losses import custom_loss
    
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(f"[warn] set_memory_growth failed: {e}")

def safe_shift_axis(x, axis, delta):
    """Shift tensor x along `axis` by `delta` with zero padding, same output shape."""
    if delta == 0:
        return x
    rank = len(x.shape)
    shp = tf.shape(x)
    paddings = [[0, 0] for _ in range(rank)]
    if delta > 0:
        paddings[axis] = [delta, 0]
        xp = tf.pad(x, paddings)
        begin = [0] * rank
        size = tf.concat([shp[:axis], [shp[axis]], shp[axis+1:]], axis=0)
        return tf.slice(xp, begin, size)
    else:
        delta = -delta
        paddings[axis] = [0, delta]
        xp = tf.pad(x, paddings)
        begin = [0] * rank
        begin[axis] = delta
        size = tf.concat([shp[:axis], [shp[axis]], shp[axis+1:]], axis=0)
        return tf.slice(xp, begin, size)

@tf.keras.utils.register_keras_serializable(package="Custom")
class StableDetectorGraphConv(layers.Layer):
    def __init__(self, channels=24, k_det=1, k_elem=2, **kwargs):
        kwargs.pop("dtype", None)
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.k_det = int(k_det)
        self.k_elem = int(k_elem)

    def build(self, input_shape):
        c_in = int(input_shape[-1])
        self.msg_self = layers.Conv2D(self.channels, 1, activation='relu', name=f'{self.name}_self')
        self.msg_det  = layers.Conv2D(self.channels, 1, activation='relu', name=f'{self.name}_det')
        self.msg_elem = layers.Conv2D(self.channels, 1, activation='relu', name=f'{self.name}_elem')
        self.update   = layers.Conv2D(self.channels, 1, activation='relu', name=f'{self.name}_upd')
        self.out_proj = layers.Conv2D(c_in, 1, name=f'{self.name}_out')
        super().build(input_shape)

    def call(self, x):
        m_self = self.msg_self(x)
        det_msgs = []
        for d in range(-self.k_det, self.k_det + 1):
            if d != 0:
                det_msgs.append(safe_shift_axis(x, axis=1, delta=d))
        if det_msgs:
            det_mean = tf.reduce_mean(tf.stack(det_msgs, axis=-1), axis=-1)
            m_det = self.msg_det(det_mean)
        else:
            m_det = tf.zeros_like(m_self)
        elem_msgs = []
        for e in range(-self.k_elem, self.k_elem + 1):
            if e != 0:
                elem_msgs.append(safe_shift_axis(x, axis=2, delta=e))
        if elem_msgs:
            elem_mean = tf.reduce_mean(tf.stack(elem_msgs, axis=-1), axis=-1)
            m_elem = self.msg_elem(elem_mean)
        else:
            m_elem = tf.zeros_like(m_self)
        h = self.update(m_self + m_det + m_elem)
        return self.out_proj(h) + x

@tf.keras.utils.register_keras_serializable(package="Custom")
class LayerScale(layers.Layer):
    def __init__(self, init_scale=0.1, **kwargs):
        kwargs.pop("dtype", None)
        super().__init__(**kwargs)
        self.init_scale = float(init_scale)

    def build(self, input_shape):
        ish = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        ch = int(ish[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(ch,),
            initializer=tf.keras.initializers.Constant(self.init_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x, delta = inputs
        return x + delta * tf.reshape(self.gamma, [1, 1, 1, -1])

@tf.keras.utils.register_keras_serializable(package="Custom")
class LightweightGNN(layers.Layer):
    def __init__(self, gnn_channels=24, num_layers=1, k_det=1, k_elem=2, **kwargs):
        kwargs.pop("dtype", None)
        super().__init__(**kwargs)
        self.gnn_channels = int(gnn_channels)
        self.num_layers = int(num_layers)
        self.k_det = int(k_det)
        self.k_elem = int(k_elem)

    def build(self, input_shape):
        c_in = int(input_shape[-1])
        self.in_proj = layers.Conv2D(self.gnn_channels, 1, activation='relu', name=f'{self.name}_in')
        self.layers_ = [
            StableDetectorGraphConv(self.gnn_channels, k_det=self.k_det, k_elem=self.k_elem, name=f'{self.name}_mp{i}')
            for i in range(self.num_layers)
        ]
        self.out_proj = layers.Conv2D(c_in, 1, name=f'{self.name}_out')
        self.ls = LayerScale(init_scale=0.1, name=f'{self.name}_ls')
        super().build(input_shape)

    def call(self, x):
        h = self.in_proj(x)
        for mp in self.layers_:
            h = mp(h)
        h = self.out_proj(h)
        return self.ls([x, h])

def unet_block(x, filters, l2=1e-4, use_bn=False, drop=0.0):
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def up_block(x, skip, filters, use_bn=False, l2=1e-4, drop=0.0):
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x

def build_model(
    num_detectors: int,
    num_elementIDs: int,
    base_filters: int = 48,
    use_bn: bool = False,
    gnn_channels: int = 24,
    gnn_layers: int = 1,
):
    D, E = int(num_detectors), int(num_elementIDs)
    inp = layers.Input(shape=(D, E, 1))
    n_down = 4
    stride = 2 ** n_down
    Dp = stride * math.ceil(D / stride)
    Ep = stride * math.ceil(E / stride)
    padD = (Dp - D)
    padE = (Ep - E)
    pad = ((padD // 2, padD - padD // 2), (padE // 2, padE - padE // 2))
    x = layers.ZeroPadding2D(padding=pad)(inp)
    f1, f2, f3, f4 = base_filters, base_filters * 2, base_filters * 4, base_filters * 5
    e1 = unet_block(x, f1, use_bn=use_bn)
    p1 = layers.MaxPooling2D(2)(e1)
    e2 = unet_block(p1, f2, use_bn=use_bn)
    e2 = LightweightGNN(gnn_channels=gnn_channels, num_layers=gnn_layers, k_det=1, k_elem=2, name='gnn_e2')(e2)
    p2 = layers.MaxPooling2D(2)(e2)
    e3 = unet_block(p2, f3, use_bn=use_bn)
    e3 = LightweightGNN(gnn_channels=gnn_channels, num_layers=gnn_layers, k_det=1, k_elem=2, name='gnn_e3')(e3)
    p3 = layers.MaxPooling2D(2)(e3)
    e4 = unet_block(p3, f4, use_bn=use_bn, drop=0.05)
    p4 = layers.MaxPooling2D(2)(e4)
    bottleneck = unet_block(p4, f4, use_bn=use_bn, drop=0.05)
    d1 = up_block(bottleneck, e4, f4, use_bn=use_bn, drop=0.0)
    d2 = up_block(d1, e3, f3, use_bn=use_bn, drop=0.0)
    d3 = up_block(d2, e2, f2, use_bn=use_bn, drop=0.0)
    d4 = up_block(d3, e1, f1, use_bn=use_bn, drop=0.0)
    d4 = layers.Cropping2D(cropping=pad)(d4)
    logits = layers.Conv2D(2, 1, name='logits')(d4)
    x = layers.Softmax(axis=2, name='softmax')(logits)
    out = layers.Permute((3, 1, 2), name='permute_out')(x)
    return tf.keras.Model(inputs=inp, outputs=out, name='TrackFinder_UGNN_v3')

def acc_exact(y_true, y_pred):
    pred_idx = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, pred_idx.dtype)
    match = tf.equal(pred_idx, y_true)
    return tf.reduce_mean(tf.cast(match, tf.float32))

def make_acc_within_k(k=2):
    def _metric(y_true, y_pred):
        pred_idx = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, pred_idx.dtype)
        diff = tf.abs(pred_idx - y_true)
        return tf.reduce_mean(tf.cast(diff <= k, tf.float32))
    _metric.__name__ = f"acc_within{k}"
    return _metric

acc_within2 = make_acc_within_k(2)

def train(
    train_root_file: str,
    val_root_file: str,
    out_path: str = "checkpoints/track_finder_UGNN_v3.h5",
    lr: float = 7e-5,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 16,
    use_bn: bool = False,
    base_filters: int = 48,
    gnn_channels: int = 24,
    gnn_layers: int = 1,
    mixed_precision: bool = False,
):
    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        print("[info] mixed precision enabled")

    X_tr, y_p_tr, y_m_tr = load_data(train_root_file)
    if X_tr is None:
        print("[error] failed to load train data.")
        return
    y_tr = np.stack([y_p_tr, y_m_tr], axis=1)

    X_va, y_p_va, y_m_va = load_data(val_root_file)
    if X_va is None:
        print("[error] failed to load val data.")
        return
    y_va = np.stack([y_p_va, y_m_va], axis=1)

    print(f"[data] X_tr={X_tr.shape}  y_tr={y_tr.shape}")
    print(f"[data] X_va={X_va.shape}  y_va={y_va.shape}")

    model = build_model(
        num_detectors=X_tr.shape[1],
        num_elementIDs=X_tr.shape[2],
        base_filters=base_filters,
        use_bn=use_bn,
        gnn_channels=gnn_channels,
        gnn_layers=gnn_layers,
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=custom_loss,
        metrics=[acc_exact, acc_within2],
        jit_compile=False
    )
    print(f"[model] params: {model.count_params():,}")

    cbs = [
        ReduceLROnPlateau(
            monitor='val_acc_within2', mode='max',
            factor=0.5, patience=4, min_delta=0.002,
            cooldown=2, min_lr=1e-7, verbose=1
        ),
        EarlyStopping(
            monitor='val_acc_within2', mode='max',
            patience=patience, min_delta=0.002,
            restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    if 'acc_within2' in history.history:
        plt.plot(history.history['acc_within2'], label='train_acc_within2')
    if 'val_acc_within2' in history.history:
        plt.plot(history.history['val_acc_within2'], label='val_acc_within2')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.legend(); plt.title('TrackFinder_UGNN_v3')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "unet_gnn_slim_losses.png"), dpi=140)
    plt.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"[save] {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet+GNN (slim, stable).")
    parser.add_argument("train_root_file", type=str)
    parser.add_argument("val_root_file", type=str)
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder_UGNN_v3.h5")
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--bn", type=int, default=0)
    parser.add_argument("--base_filters", type=int, default=48)
    parser.add_argument("--gnn_channels", type=int, default=24)
    parser.add_argument("--gnn_layers", type=int, default=1)
    parser.add_argument("--mixed_precision", type=int, default=0)
    args = parser.parse_args()

    train(
        args.train_root_file,
        args.val_root_file,
        out_path=args.output_model,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch,
        use_bn=bool(args.bn),
        base_filters=args.base_filters,
        gnn_channels=args.gnn_channels,
        gnn_layers=args.gnn_layers,
        mixed_precision=bool(args.mixed_precision),
    )
