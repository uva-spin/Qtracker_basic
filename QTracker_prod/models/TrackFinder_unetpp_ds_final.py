
import os, math, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("checkpoints", exist_ok=True)


def load_data(root_file):
    from data_loader import load_data as _load
    return _load(root_file)


def _cast_compute(x, name=None):
    return layers.Activation("linear",
        dtype=mixed_precision.global_policy().compute_dtype, name=name)(x)

def parse_band_weight(s: str, D: int):
    w = np.ones([D], dtype=np.float32)
    if s:
        try:
            rng, val = s.split(":"); a, b = rng.split("-")
            a, b = int(a), int(b); val = float(val)
            a = max(1, min(D, a)); b = max(1, min(D, b))
            if a > b: a, b = b, a
            w[a-1:b] = val
        except Exception:
            pass
    return tf.constant(w, tf.float32)

def parse_mask_ranges(s: str, D: int):
    m = np.ones([D], dtype=np.float32)
    if s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            if "-" in p:
                a, b = p.split("-"); a = int(a); b = int(b)
                a = max(1, min(D, a)); b = max(1, min(D, b))
                if a > b: a, b = b, a
                m[a-1:b] = 0.0
            else:
                i = int(p)
                if 1 <= i <= D: m[i-1] = 0.0
    return tf.constant(m, tf.float32)

@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalChannels(layers.Layer):
    def call(self, x):
        shp = tf.shape(x); B, Dp, Ep = shp[0], shp[1], shp[2]
        det = tf.cast(tf.range(Dp), tf.float32) / tf.maximum(1.0, tf.cast(Dp-1, tf.float32))
        ele = tf.cast(tf.range(Ep), tf.float32) / tf.maximum(1.0, tf.cast(Ep-1, tf.float32))
        det_ch = tf.tile(det[:, None], [1, Ep])
        ele_ch = tf.tile(ele[None, :], [Dp, 1])
        r  = tf.sqrt(tf.square(det_ch - 0.5) + tf.square(ele_ch - 0.5))
        r2 = tf.square(r)
        pos = tf.stack([det_ch, ele_ch, r, r2], axis=-1)   # (D,E,4)
        pos = tf.expand_dims(pos, 0)                       # (1,D,E,4)
        return tf.tile(pos, [B,1,1,1])                     # (B,D,E,4)

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMeanC(layers.Layer):
    def call(self, x): return tf.reduce_mean(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxC(layers.Layer):
    def call(self, x): return tf.reduce_max(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable(package="Custom")
class ToSeqDet(layers.Layer):
    def call(self, t):
        return tf.reshape(tf.transpose(t, [0,2,1,3]),
                          [-1, tf.shape(t)[1], tf.shape(t)[3]])

@tf.keras.utils.register_keras_serializable(package="Custom")
class FromSeqDet(layers.Layer):
    def call(self, inputs):
        seq, ref = inputs
        B = tf.shape(ref)[0]; D = tf.shape(ref)[1]; E = tf.shape(ref)[2]; C = tf.shape(ref)[3]
        x = tf.reshape(seq, [B, E, D, C])
        return tf.transpose(x, [0,2,1,3])

@tf.keras.utils.register_keras_serializable(package="Custom")
class StopGradient(layers.Layer):
    def call(self, x):
        return tf.stop_gradient(x)

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxLast(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1)
    def compute_output_shape(self, input_shape):
        # input_shape: (B,D,E,C) -> output: (B,D,E)
        return input_shape[:-1]


def conv_block(x, filters, l2=1e-4, use_bn=False, dropout_bn=0.0, dropout=0.0, name=None):
    sc = x
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2),
                      name=None if name is None else f"{name}_c1")(x)
    if use_bn: x = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(x)
    x = _cast_compute(x); x = layers.Activation("relu")(x)

    if dropout_bn > 0: x = layers.Dropout(dropout_bn)(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=regularizers.l2(l2),
                      name=None if name is None else f"{name}_c2")(x)
    if use_bn: x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)

    # CBAM
    gap = layers.GlobalAveragePooling2D(keepdims=True)(x)
    gmp = layers.GlobalMaxPooling2D(keepdims=True)(x)
    rc  = max(1, filters // 8)
    mlp1 = layers.Conv2D(rc, 1, activation="relu", padding="same")
    mlp2 = layers.Conv2D(filters, 1, activation=None, padding="same")
    ca = layers.Activation("sigmoid")(layers.Add()([mlp2(mlp1(gap)), mlp2(mlp1(gmp))]))
    x = layers.Multiply()([x, ca])
    avg = ReduceMeanC()(x); mx = ReduceMaxC()(x)
    sa  = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(layers.Concatenate(axis=-1)([avg, mx]))
    x   = layers.Multiply()([x, sa])

    if sc.shape[-1] != x.shape[-1]:
        sc = layers.Conv2D(filters, 1, padding="same")(sc)
    x = _cast_compute(x); sc = _cast_compute(sc)
    x = layers.Add()([x, sc])
    x = _cast_compute(x); x = layers.Activation("relu")(x)
    if dropout > 0: x = layers.Dropout(dropout)(x)
    return x

def up2x(x): 
    return layers.UpSampling2D(interpolation="bilinear")(x)

def axial_det_block(x, num_heads=4, key_dim=32, dropout=0.0, ff_ratio=2.0, name=None):
    C  = int(x.shape[-1])
    ln1 = layers.LayerNormalization(epsilon=1e-6)(x)
    seq = ToSeqDet()(ln1)                     # (B*E,D,C)
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(seq, seq)
    att = FromSeqDet()([att, ln1])            # (B,D,E,C)
    add1= layers.Add()([_cast_compute(x), _cast_compute(att)])
    ln2 = layers.LayerNormalization(epsilon=1e-6)(add1)
    seq2= ToSeqDet()(ln2)
    hid = max(1, int(ff_ratio*C))
    z   = layers.Conv1D(hid, 1, activation="gelu")(seq2)
    z   = layers.Dropout(dropout)(z)
    z   = layers.Conv1D(C,   1)(z)
    z   = layers.Dropout(dropout)(z)
    z   = FromSeqDet()([z, ln2])
    return layers.Add()([_cast_compute(add1), _cast_compute(z)])

def det_tcn_block(x, kernel=5, dilations=(1,2,4,8), dropout=0.05, name=None):
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    for d in dilations:
        z = layers.DepthwiseConv2D((kernel,1), dilation_rate=(d,1), padding="same")(h)
        z = layers.Conv2D(int(x.shape[-1]), 1, activation="gelu")(z)
        g = layers.Conv2D(int(x.shape[-1]), 1, activation="sigmoid")(h)
        z = layers.Multiply()([_cast_compute(g), _cast_compute(z)])
        z = layers.Dropout(dropout)(z)
        h = layers.Add()([h, z])
    out = layers.Conv2D(int(x.shape[-1]), 1)(h)
    return layers.Add()([_cast_compute(x), _cast_compute(out)])

def build_denoiser(D=62, E=201, base=32, use_bn=True, dropout=0.0, l2=1e-4):
    inp = layers.Input(shape=(D,E,1), name="den_in")
    depth=4; num_pool=2**depth
    Dp = num_pool * math.ceil(D/num_pool); Ep = num_pool * math.ceil(E/num_pool)
    ddf, edf = Dp-D, Ep-E; pad = ((ddf//2, ddf - ddf//2), (edf//2, edf - edf//2))
    x = layers.ZeroPadding2D(padding=pad, name="den_pad")(inp)

    c1 = conv_block(x,  base,    use_bn=use_bn, dropout=dropout, l2=l2); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base*2,  use_bn=use_bn, dropout=dropout, l2=l2); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base*4,  use_bn=use_bn, dropout=dropout, l2=l2); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base*8,  use_bn=use_bn, dropout=dropout, l2=l2); p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, base*16, use_bn=use_bn, dropout=dropout, l2=l2)

    u4 = up2x(bn); u4 = layers.Concatenate()([u4, c4]); c5 = conv_block(u4, base*8,  use_bn=use_bn, l2=l2)
    u3 = up2x(c5); u3 = layers.Concatenate()([u3, c3]); c6 = conv_block(u3, base*4,  use_bn=use_bn, l2=l2)
    u2 = up2x(c6); u2 = layers.Concatenate()([u2, c2]); c7 = conv_block(u2, base*2,  use_bn=use_bn, l2=l2)
    u1 = up2x(c7); u1 = layers.Concatenate()([u1, c1]); c8 = conv_block(u1, base,    use_bn=use_bn, l2=l2)
    if dropout>0: c8 = layers.Dropout(dropout)(c8)

    full = layers.Conv2D(2, 1, activation="sigmoid", name="den_out_full")(c8) 
    full = _cast_compute(full)
    out  = layers.Cropping2D(cropping=pad, name="deno_2ch")(full)   
    return tf.keras.Model(inp, out, name="DenoiserUNet")

def _head_logits(feature, cropping):
    x = layers.Cropping2D(cropping=cropping)(feature)
    return layers.Conv2D(2, 1)(x)

def build_model(num_detectors=62, num_elementIDs=201,
                base=64, use_bn=True, dropout_bn=0.10, dropout_enc=0.05, l2=1e-4,
                den_base=32, den_bn=True, den_dropout=0.05,
                axial_heads=4, axial_keydim=32, axial_drop=0.0,
                tcn_kernel=5, tcn_dils=(1,2,4,8), tcn_drop=0.05):
    D=num_detectors; E=num_elementIDs
    inp = layers.Input(shape=(D,E,1), name="input")

    deno = build_denoiser(D,E,base=den_base,use_bn=den_bn,dropout=den_dropout)(inp)  

    deno_detached = StopGradient(name="deno_detached")(deno)

    pos  = PositionalChannels(name="positional_channels")(inp)
    x_in = layers.Concatenate(name="concat_input")([inp, deno_detached, pos]) 

    filters = [base, base*2, base*4, base*8, base*16]
    depth = len(filters) - 1
    num_pool = 2 ** depth
    Dp = num_pool * math.ceil(D / num_pool)
    Ep = num_pool * math.ceil(E / num_pool)
    ddf, edf = Dp - D, Ep - E
    padding = ((ddf // 2, ddf - ddf // 2), (edf // 2, edf - edf // 2))
    x0 = layers.ZeroPadding2D(padding=padding, name="pad")(x_in)

    X = [[None] * len(filters) for _ in range(len(filters))]
    X[0][0] = conv_block(x0, filters[0], use_bn=use_bn, dropout_bn=dropout_bn, name="enc0_0")
    p1 = layers.MaxPooling2D(name="pool1")(X[0][0])
    X[1][0] = conv_block(p1, filters[1], use_bn=use_bn, name="enc1_0"); p2 = layers.MaxPooling2D(name="pool2")(X[1][0])
    X[2][0] = conv_block(p2, filters[2], use_bn=use_bn, dropout=dropout_enc, name="enc2_0"); p3 = layers.MaxPooling2D(name="pool3")(X[2][0])
    X[3][0] = conv_block(p3, filters[3], use_bn=use_bn, dropout=dropout_enc, name="enc3_0"); p4 = layers.MaxPooling2D(name="pool4")(X[3][0])
    X[4][0] = conv_block(p4, filters[4], use_bn=use_bn, dropout_bn=dropout_bn, name="enc4_0")

    for j in range(1, len(filters)):
        for i in range(0, len(filters) - j):
            parts = [X[i][k] for k in range(j)] + [up2x(X[i+1][j-1])]
            z = layers.Concatenate(name=f"cat_{i}_{j}")(parts)
            z = _cast_compute(z)
            X[i][j] = conv_block(z, filters[i], use_bn=use_bn, name=f"dec{i}_{j}")

    feat = X[0][depth]
    feat = axial_det_block(feat, num_heads=axial_heads, key_dim=axial_keydim, dropout=axial_drop, name="axdet_final")
    feat = det_tcn_block(feat, kernel=tcn_kernel, dilations=tcn_dils, dropout=tcn_drop, name="tcn_final")

    logits = _head_logits(feat, padding)      
    probs  = layers.Softmax(axis=2, dtype="float32", name="softmax_over_E")(logits)
    pred   = layers.Permute((3, 1, 2), name="pred")(probs)  

    deno_map = ReduceMaxLast(name="deno_map")(deno)
    
    return tf.keras.Model(
        inputs=inp,
        outputs={"pred": pred, "deno_map": deno_map},
        name="TrackFinder_UNetPP_noDS_dual"
    )

def _smoothed_sparse_ce(y_idx, p_dist, eps):
    y_idx = tf.cast(y_idx, tf.int32)
    p_dist = tf.cast(p_dist, tf.float32)
    E = tf.shape(p_dist)[2]
    oh = tf.one_hot(y_idx, depth=E, dtype=tf.float32)
    if eps>0: oh = (1.0 - eps)*oh + eps/tf.cast(E, tf.float32)
    return tf.keras.losses.categorical_crossentropy(oh, p_dist)

def _gauss_ce(y_idx, p_dist, sigma):
    y_idx = tf.cast(y_idx, tf.float32); p_dist = tf.cast(p_dist, tf.float32)
    E = tf.shape(p_dist)[2]; x = tf.cast(tf.range(E)[None,None,:], tf.float32)
    g = tf.exp(-0.5*tf.square((x - y_idx[...,None]) / sigma))
    g = g / (tf.reduce_sum(g, axis=-1, keepdims=True) + 1e-8)
    return tf.keras.losses.categorical_crossentropy(g, p_dist)

def _expected_index(p_dist):
    p_dist = tf.cast(p_dist, tf.float32)
    E = tf.shape(p_dist)[2]
    pos = tf.cast(tf.range(E)[None,None,:], tf.float32)
    return tf.reduce_sum(p_dist * pos, axis=-1)

def make_tracking_loss(E, band_weight_vec, ls_eps=0.05, gauss_lambda=0.05, gauss_sigma=1.0,
                       overlap_lambda=0.05, smooth_lambda=0.03, smooth_k=4.0, smooth_mask=None):
    band_w = tf.reshape(tf.cast(band_weight_vec, tf.float32), [1,-1])  # (1,D)
    if smooth_mask is None: smooth_mask = tf.ones_like(band_w[0], dtype=tf.float32)
    smooth_mask = tf.cast(smooth_mask, tf.float32)
    k = tf.constant(smooth_k, tf.float32); eps = tf.constant(1e-4, tf.float32)

    def loss(y_true, y_pred):
        y_mup_t, y_mum_t = tf.unstack(y_true, num=2, axis=1)         
        y_mup_p = y_pred[:,0,:,:]; y_mum_p = y_pred[:,1,:,:]        # (B,D,E)
        ce_mup = _smoothed_sparse_ce(y_mup_t, y_mup_p, ls_eps)
        ce_mum = _smoothed_sparse_ce(y_mum_t, y_mum_p, ls_eps)
        if gauss_lambda>0:
            ce_mup += gauss_lambda * _gauss_ce(y_mup_t, y_mup_p, gauss_sigma)
            ce_mum += gauss_lambda * _gauss_ce(y_mum_t, y_mum_p, gauss_sigma)
        ce_mup *= band_w; ce_mum *= band_w

        ce_ol = 0.0
        if overlap_lambda>0:
            ol = tf.reduce_sum(tf.sqrt(y_mup_p*y_mum_p + 1e-9), axis=-1)
            ce_ol = overlap_lambda * ol

        sm = 0.0
        if smooth_lambda>0:
            mu_p = _expected_index(y_mup_p); mu_m = _expected_index(y_mum_p)  
            valid_pair = smooth_mask[1:] * smooth_mask[:-1]
            valid_pair = tf.reshape(valid_pair, [1, -1])  
            def tv_hinge(mu):
                d = tf.abs(mu[:,1:] - mu[:,:-1])
                charb = tf.sqrt(d*d + eps) - tf.sqrt(k*k + eps)
                pen = tf.nn.relu(charb) * valid_pair
                return tf.reduce_mean(pen)
            sm = smooth_lambda * (tv_hinge(mu_p) + tv_hinge(mu_m))
        return tf.reduce_mean(ce_mup + ce_mum + ce_ol) + sm
    return loss

def metric_top1_mean_acc(y_true, y_pred):
    y_mup_t, y_mum_t = tf.split(y_true, 2, axis=1)
    y_mup_t = tf.squeeze(y_mup_t, axis=1); y_mum_t = tf.squeeze(y_mum_t, axis=1)
    mup_arg = tf.argmax(y_pred[:,0,:,:], axis=-1, output_type=tf.int32)
    mum_arg = tf.argmax(y_pred[:,1,:,:], axis=-1, output_type=tf.int32)
    a = tf.cast(tf.equal(mup_arg, y_mup_t), tf.float32)
    b = tf.cast(tf.equal(mum_arg, y_mum_t), tf.float32)
    return tf.reduce_mean(0.5*(a+b))

def metric_distance_acc_k2(y_true, y_pred):
    y_mup_t, y_mum_t = tf.split(y_true, 2, axis=1)
    y_mup_t = tf.squeeze(y_mup_t, axis=1); y_mum_t = tf.squeeze(y_mum_t, axis=1)
    mup_arg = tf.argmax(y_pred[:,0,:,:], axis=-1, output_type=tf.int32)
    mum_arg = tf.argmax(y_pred[:,1,:,:], axis=-1, output_type=tf.int32)
    a = tf.abs(tf.cast(mup_arg, tf.int32) - tf.cast(y_mup_t, tf.int32)) <= 2
    b = tf.abs(tf.cast(mum_arg, tf.int32) - tf.cast(y_mum_t, tf.int32)) <= 2
    return tf.reduce_mean(tf.cast(a, tf.float32)*0.5 + tf.cast(b, tf.float32)*0.5)

def metric_core_ce(y_true, y_pred):
    y_mup_t, y_mum_t = tf.split(y_true, 2, axis=1)
    y_mup_t = tf.squeeze(y_mup_t, axis=1); y_mum_t = tf.squeeze(y_mum_t, axis=1)
    ce = _smoothed_sparse_ce(y_mup_t, y_pred[:,0,:,:], 0.0) + _smoothed_sparse_ce(y_mum_t, y_pred[:,1,:,:], 0.0)
    return tf.reduce_mean(0.5*ce)


def resolve_output_keys(model):
    names = list(model.output_names)
    pred_key = "pred" if "pred" in names else names[0]
    if "deno_map" in names:
        den_key = "deno_map"
    elif "DenoiserUNet" in names:
        den_key = "DenoiserUNet"
    else:
        den_candidates = [n for n in names if n != pred_key]
        den_key = den_candidates[0] if den_candidates else names[-1]
    return pred_key, den_key


def _compile_with_new_optimizer(model, lr, wd, clip, loss_pred, deno_weight, pred_key, den_key):
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=clip)
    model.compile(
        optimizer=opt,
        loss={
            pred_key: loss_pred,
            den_key: tf.keras.losses.BinaryCrossentropy(from_logits=False),
        },
        loss_weights={pred_key: 1.0, den_key: float(deno_weight)},
        metrics={
            pred_key: [metric_top1_mean_acc, metric_distance_acc_k2, metric_core_ce],
            den_key:  [tf.keras.metrics.AUC(curve="ROC", name="den_auc")],
        },
    )

def _run_stage(model, X_tr, y_tr, X_val, y_val, stage_name, max_epochs, patience,
               lr, wd, clip, deno_weight, global_saver, basepath, batch_size,
               pred_key, den_key):
    _compile_with_new_optimizer(model, lr, wd, clip, model.loss_pred, deno_weight, pred_key, den_key)

    stage_ckpt_path = basepath + f"_{stage_name}.best.weights.h5"
    ckpt_stage = tf.keras.callbacks.ModelCheckpoint(
        filepath=stage_ckpt_path, monitor="val_loss", mode="min",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    rlop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=max(1, patience//3), min_lr=1e-6, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    if max_epochs > 0:
        print(f"[STAGE {stage_name}] epochs={max_epochs}, batch_size={batch_size}, lr={lr:.6g}, deno_w={deno_weight}")
        model.fit(
            X_tr, {pred_key: y_tr, den_key: X_tr[..., 0]},
            epochs=max_epochs,
            batch_size=batch_size,
            validation_data=(X_val, {pred_key: y_val, den_key: X_val[..., 0]}),
            callbacks=[ckpt_stage, rlop, es, global_saver],
            verbose=1, shuffle=True,
        )

    if os.path.exists(stage_ckpt_path):
        model.load_weights(stage_ckpt_path)

class GlobalBestSaver(tf.keras.callbacks.Callback):
    def __init__(self, basepath):
        super().__init__()
        self.base = basepath
        self.best_loss = float("inf")
        self.best_acc  = -float("inf")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vloss = logs.get("val_loss")
        vacc  = (logs.get("val_pred_metric_top1_mean_acc")
                 or logs.get("val_metric_top1_mean_acc"))
        if vloss is not None and vloss < self.best_loss - 1e-12:
            self.best_loss = vloss
            self.model.save_weights(self.base + "_val_loss.weights.h5")
        if vacc is not None and vacc > self.best_acc + 1e-12:
            self.best_acc = vacc
            self.model.save_weights(self.base + "_val_accuracy.weights.h5")

def train_model(args):
    X_low,  y_mup_l, y_mum_l  = load_data(args.train_root_file_low)
    X_med,  y_mup_m, y_mum_m  = load_data(args.train_root_file_med)
    X_high, y_mup_h, y_mum_h  = load_data(args.train_root_file_high)
    X_val,  y_mup_v, y_mum_v  = load_data(args.val_root_file)
    if any(v is None for v in [X_low, X_med, X_high, X_val]):
        print("[ERROR] Failed to load one or more ROOT files."); return

    y_low  = np.stack([y_mup_l, y_mum_l], axis=1).astype("int32")
    y_med  = np.stack([y_mup_m, y_mum_m], axis=1).astype("int32")
    y_high = np.stack([y_mup_h, y_mum_h], axis=1).astype("int32")
    y_val  = np.stack([y_mup_v, y_mum_v], axis=1).astype("int32")

    D, E = X_low.shape[1], X_low.shape[2]
    model = build_model(
        num_detectors=D, num_elementIDs=E,
        base=args.base, use_bn=bool(args.batch_norm),
        dropout_bn=args.dropout_bn, dropout_enc=args.dropout_enc, l2=1e-4,
        den_base=args.den_base, den_bn=bool(args.batch_norm), den_dropout=args.den_dropout,
        axial_heads=args.axial_heads, axial_keydim=args.axial_keydim, axial_drop=args.axial_dropout,
        tcn_kernel=args.tcn_kernel, tcn_dils=tuple(args.tcn_dilations), tcn_drop=args.tcn_dropout
    )
    print(model.summary())

    pred_key, den_key = resolve_output_keys(model)
    print(f"[INFO] model.output_names = {model.output_names}")
    print(f"[INFO] Using keys -> pred_key='{pred_key}', den_key='{den_key}'")

    band = parse_band_weight(args.muplus_band, D)
    overlap_lambda = float(args.overlap_band.split(":")[1]) if args.overlap_band else 0.0
    smooth_mask = parse_mask_ranges(args.smooth_mask_ranges, D)
    track_loss = make_tracking_loss(
        E=E, band_weight_vec=band, ls_eps=args.label_smoothing_eps,
        gauss_lambda=args.gauss_lambda, gauss_sigma=args.gauss_sigma,
        overlap_lambda=overlap_lambda, smooth_lambda=args.smooth_lambda, smooth_k=args.smooth_k,
        smooth_mask=smooth_mask
    )
    model.loss_pred = track_loss

    epochs_low = int(args.epochs * args.low_ratio)
    epochs_med = int(args.epochs * args.med_ratio)
    epochs_high= args.epochs

    basepath = os.path.splitext(args.output_model)[0]
    global_saver = GlobalBestSaver(basepath)

    bs_high = args.bs_high if args.bs_high is not None else args.batch_size
    bs_med  = args.bs_med  if args.bs_med  is not None else int(round(1.5 * args.batch_size))
    bs_low  = args.bs_low  if args.bs_low  is not None else 3 * args.batch_size
    lr_high = args.lr_high if args.lr_high is not None else args.learning_rate
    lr_med  = args.lr_med  if args.lr_med  is not None else args.learning_rate * 1.5
    lr_low  = args.lr_low  if args.lr_low  is not None else args.learning_rate * 2.0

    # ---- Stage: LOW ----  (deno_weight = 0.20, trainable=True)
    _run_stage(model, X_low, y_low, X_val, y_val,
        stage_name="LOW", max_epochs=epochs_low, patience=args.patience,
        lr=lr_low, wd=args.weight_decay, clip=args.clipnorm,
        deno_weight=0.20,
        global_saver=global_saver, basepath=basepath, batch_size=bs_low,
        pred_key=pred_key, den_key=den_key,
    )
    
    # ---- Stage: MED ----  (deno_weight = 0.05, Denoiser freeze)
    model.get_layer("DenoiserUNet").trainable = False
    _run_stage(model, X_med, y_med, X_val, y_val,
        stage_name="MED", max_epochs=max(0, epochs_med - epochs_low), patience=args.patience,
        lr=lr_med, wd=args.weight_decay, clip=args.clipnorm,
        deno_weight=0.05,
        global_saver=global_saver, basepath=basepath, batch_size=bs_med,
        pred_key=pred_key, den_key=den_key,
    )
    
    # ---- Stage: HIGH ---- (deno_weight = 0.00, Denoiser freeze)
    _run_stage(model, X_high, y_high, X_val, y_val,
        stage_name="HIGH", max_epochs=max(0, epochs_high - epochs_med), patience=args.patience,
        lr=lr_high, wd=args.weight_decay, clip=args.clipnorm,
        deno_weight=0.00,
        global_saver=global_saver, basepath=basepath, batch_size=bs_high,
        pred_key=pred_key, den_key=den_key,
    )

    if os.path.exists(basepath + "_val_loss.weights.h5"):
        model.load_weights(basepath + "_val_loss.weights.h5")
        pred_only = tf.keras.Model(model.input, model.get_layer("pred").output, name=model.name + "_single")
        pred_only.save(basepath + "_val_loss.h5", include_optimizer=False)
        print(f"[INFO] Saved: {basepath}_val_loss.h5 (single-head)")
    else:
        print("[WARN] global best(val_loss) weights not found; skip export")
    
    if os.path.exists(basepath + "_val_accuracy.weights.h5"):
        model.load_weights(basepath + "_val_accuracy.weights.h5")
        pred_only = tf.keras.Model(model.input, model.get_layer("pred").output, name=model.name + "_single")
        pred_only.save(basepath + "_val_accuracy.h5", include_optimizer=False)
        print(f"[INFO] Saved: {basepath}_val_accuracy.h5 (single-head)")
    else:
        print("[WARN] global best(val_accuracy) weights not found; skip export")


def main():
    ap = argparse.ArgumentParser(description="UNet++ no-DS + Denoise/Axial/TCN/CBAM (dual-output) with Curriculum Learning")
    ap.add_argument("train_root_file_low",  type=str)
    ap.add_argument("train_root_file_med",  type=str)
    ap.add_argument("train_root_file_high", type=str)
    ap.add_argument("val_root_file",        type=str)
    ap.add_argument("--output_model",  type=str, default="checkpoints/track_finder_unetpp_nods_dual.h5")

    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--patience",      type=int,   default=12)
    ap.add_argument("--epochs",        type=int,   default=160)
    ap.add_argument("--batch_size",    type=int,   default=32)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--clipnorm",      type=float, default=1.0)
    ap.add_argument("--deno_weight",   type=float, default=0.10)

    ap.add_argument("--low_ratio",     type=float, default=0.5)
    ap.add_argument("--med_ratio",     type=float, default=0.8)

    ap.add_argument("--lr_low",  type=float, default=None)
    ap.add_argument("--lr_med",  type=float, default=None)
    ap.add_argument("--lr_high", type=float, default=None)
    ap.add_argument("--bs_low",  type=int,   default=None)
    ap.add_argument("--bs_med",  type=int,   default=None)
    ap.add_argument("--bs_high", type=int,   default=None)

    ap.add_argument("--batch_norm",    type=int,   default=1)
    ap.add_argument("--dropout_bn",    type=float, default=0.10)
    ap.add_argument("--dropout_enc",   type=float, default=0.05)
    ap.add_argument("--base",          type=int,   default=64)

    ap.add_argument("--den_base",      type=int,   default=32)
    ap.add_argument("--den_dropout",   type=float, default=0.05)

    ap.add_argument("--axial_heads",   type=int,   default=4)
    ap.add_argument("--axial_keydim",  type=int,   default=32)
    ap.add_argument("--axial_dropout", type=float, default=0.0)
    ap.add_argument("--tcn_kernel",    type=int,   default=5)
    ap.add_argument("--tcn_dilations", nargs="+",  type=int, default=[1,2,4,8])
    ap.add_argument("--tcn_dropout",   type=float, default=0.05)

    ap.add_argument("--label_smoothing_eps", type=float, default=0.05)
    ap.add_argument("--gauss_lambda",  type=float, default=0.05)
    ap.add_argument("--gauss_sigma",   type=float, default=1.0)
    ap.add_argument("--muplus_band",   type=str,   default="")
    ap.add_argument("--overlap_band",  type=str,   default="")
    ap.add_argument("--smooth_lambda", type=float, default=0.03)
    ap.add_argument("--smooth_k",      type=float, default=4.0)
    ap.add_argument("--smooth_mask_ranges", type=str, default="7-12,55-62")

    args = ap.parse_args()
    tf.keras.backend.clear_session()
    train_model(args)

if __name__ == "__main__":
    main()
