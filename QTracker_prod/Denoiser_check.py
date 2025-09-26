import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, mixed_precision

mixed_precision.set_global_policy("mixed_float16")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalChannels(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
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
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMeanC(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, x): return tf.reduce_mean(x, axis=-1, keepdims=True)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxC(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, x): return tf.reduce_max(x, axis=-1, keepdims=True)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ToSeqDet(layers.Layer):
    """(B,D,E,C) -> (B*E, D, C)"""
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, t):
        return tf.reshape(tf.transpose(t, [0,2,1,3]),
                          [-1, tf.shape(t)[1], tf.shape(t)[3]])
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class FromSeqDet(layers.Layer):
    """(seq=(B*E,D,C), ref=(B,D,E,C)) -> (B,D,E,C)"""
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, inputs):
        seq, ref = inputs
        B = tf.shape(ref)[0]; D = tf.shape(ref)[1]; E = tf.shape(ref)[2]; C = tf.shape(ref)[3]
        x = tf.reshape(seq, [B, E, D, C])
        return tf.transpose(x, [0,2,1,3])
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class L1MeanLoss(layers.Layer):
    def __init__(self, coef=0.0, **kwargs):
        super().__init__(**kwargs)
        self.coef = float(coef)

    def call(self, x):
        m = tf.reduce_mean(tf.cast(x, tf.float32))
        if self.coef > 0.0:
            self.add_loss(self.coef * m)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"coef": self.coef})
        return cfg

@tf.keras.utils.register_keras_serializable(package="Custom")
class StopGradient(layers.Layer):
    def call(self, x): return tf.stop_gradient(x)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxLast(layers.Layer):
    def call(self, x): return tf.reduce_max(x, axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    def get_config(self): return {}

def load_data(root_file):
    from data_loader import load_data as _load
    return _load(root_file)  # -> X, y_mup, y_mum
  
def get_denoiser_layer(model, prefer_name="DenoiserUNet"):
    try:
        return model.get_layer(prefer_name)
    except ValueError:
        candidates = [l for l in model.layers if ("Denoiser" in l.name or "den" in l.name)]
        if not candidates:
            raise ValueError("Denoiser layer not found. Check layer name.")
        return candidates[0]

def make_models_split(model):
    x_concat = model.get_layer("concat_input").output
    f_front  = tf.keras.Model(model.input, x_concat)
    pad_in = model.get_layer("pad").input
    f_back = tf.keras.Model(pad_in, model.output)
    return f_front, f_back

def _imshow_DE(ax, arr_DE, title, vmin=None, vmax=None, cmap="viridis", add_cbar=False):
    D, E = arr_DE.shape
    im = ax.imshow(arr_DE.T, extent=[0, D, 0, E], origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title); ax.set_xlabel("Detector"); ax.set_ylabel("Element")
    ax.set_xlim(0, D); ax.set_ylim(0, E)
    if add_cbar: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

def _plot_gt_tracks(ax, y_mup, y_mum, label_prefix="GT"):
    D = len(y_mup); det = np.arange(D)
    ax.scatter(det, y_mup, s=8, c="white", edgecolors="black", linewidths=0.2, label=f"{label_prefix} μ+")
    ax.scatter(det, y_mum, s=8, c="gold",  edgecolors="black", linewidths=0.2, label=f"{label_prefix} μ−")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

def plot_suite_readable(model, deno_model, X_val, y_mup_val, y_mum_val,
                        idx=0, outdir="./denoiser_vis", prefix=None,
                        pos_quantile=0.90):
    os.makedirs(outdir, exist_ok=True)
    if prefix is None: prefix = f"idx{idx}"

    x = X_val[idx:idx+1]                  # (1,D,E,1)
    x0 = x[0, :, :, 0].astype("float32")  # (D,E)
    y_mup = y_mup_val[idx].astype(int); y_mum = y_mum_val[idx].astype(int)

    deno = deno_model.predict(x, verbose=0)[0]      # (D,E,2)
    d0, d1 = deno[...,0], deno[...,1]
    dmax = np.maximum(d0, d1)
    delta = dmax - x0

    figA, axsA = plt.subplots(1, 2, figsize=(11, 4))
    _imshow_DE(axsA[0], x0, "Input hits (binary)", vmin=0, vmax=1, cmap="magma")
    _imshow_DE(axsA[1], np.zeros_like(x0), "Real hits (GT tracks)", vmin=0, vmax=1, cmap="Greys")
    _plot_gt_tracks(axsA[1], y_mup, y_mum)
    figA.tight_layout(); figA.savefig(os.path.join(outdir, f"{prefix}_A_input_vs_gt.png"), dpi=160); plt.close(figA)

    lim = np.max(np.abs(delta)) or 1.0
    figB, axB = plt.subplots(1,1, figsize=(6,4.2))
    _imshow_DE(axB, delta, "Denoiser Δ (↑ red, ↓ blue)", vmin=-lim, vmax=lim, cmap="bwr", add_cbar=True)
    figB.tight_layout(); figB.savefig(os.path.join(outdir, f"{prefix}_B_delta.png"), dpi=160); plt.close(figB)

    pos = np.clip(delta, 0, None)
    thr = np.quantile(pos[pos>0], pos_quantile) if np.any(pos>0) else 0.0
    strong_mask = (pos >= thr) & (thr > 0)

    figC, axC = plt.subplots(1,1, figsize=(7.5,4.2))
    _imshow_DE(axC, dmax, "Denoiser max(ch) + GT tracks", vmin=0, vmax=1, cmap="viridis", add_cbar=True)
    _plot_gt_tracks(axC, y_mup, y_mum)
    if strong_mask.any():
        D,E = strong_mask.shape
        axC.contour(strong_mask.T.astype(float), levels=[0.5],
                    extent=[0,D,0,E], origin="lower", linewidths=0.9, colors="white")
        axC.text(0.01, 0.98, f"emphasis ≥ Q{int(pos_quantile*100)}", transform=axC.transAxes,
                 ha="left", va="top", fontsize=9, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35, ec="none"))
    figC.tight_layout(); figC.savefig(os.path.join(outdir, f"{prefix}_C_dmax_with_gt.png"), dpi=160); plt.close(figC)

    band = 2
    D,E = x0.shape
    mask = np.zeros_like(dmax, dtype=bool)
    for d in range(D):
        mask[d, max(0,y_mup[d]-band):min(E, y_mup[d]+band+1)] = True
        mask[d, max(0,y_mum[d]-band):min(E, y_mum[d]+band+1)] = True
    on  = dmax[mask].mean() if mask.any() else float("nan")
    off = dmax[~mask].mean() if (~mask).any() else float("nan")
    print(f"[{prefix}] on/off mean: {on:.3f} / {off:.3f}  (ratio {on/(off+1e-8):.2f})")

def _auroc_from_scores(scores, labels):
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    pos = scores[labels]; neg = scores[~labels]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return np.nan
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(all_scores)) + 1  # 1-based
    sum_ranks_pos = ranks[:n1].sum()
    U = sum_ranks_pos - n1*(n1+1)/2
    return U / (n1*n0)

def dataset_denoiser_stats(deno_model, X_val, y_mup, y_mum, n=256, band=2):
    n = min(n, len(X_val))
    ratios, aucs = [], []
    for i in range(n):
        x  = X_val[i:i+1]
        D,E = x.shape[1], x.shape[2]
        den = deno_model.predict(x, verbose=0)[0]
        dmax= np.maximum(den[...,0], den[...,1])

        y1 = y_mup[i].astype(int); y2 = y_mum[i].astype(int)
        mask = np.zeros((D,E), dtype=bool)
        for d in range(D):
            mask[d, max(0,y1[d]-band):min(E,y1[d]+band+1)] = True
            mask[d, max(0,y2[d]-band):min(E,y2[d]+band+1)] = True

        on  = dmax[mask].mean() if mask.any() else np.nan
        off = dmax[~mask].mean() if (~mask).any() else np.nan
        ratios.append(on/(off+1e-8))

        scores = dmax.flatten()
        labels = mask.flatten().astype(int)
        aucs.append(_auroc_from_scores(scores, labels))
    ratios, aucs = np.array(ratios), np.array(aucs)
    print(f"[DEN] on/off ratio: mean {np.nanmean(ratios):.3f}, median {np.nanmedian(ratios):.3f}, "
          f"Q25 {np.nanquantile(ratios,0.25):.3f}, Q75 {np.nanquantile(ratios,0.75):.3f}")
    print(f"[DEN] AUROC: mean {np.nanmean(aucs):.3f}, median {np.nanmedian(aucs):.3f}")
    return ratios, aucs

def _metrics_from_pred(pred, y_mup, y_mum):
    p = np.asarray(pred, dtype=np.float32)
    p = np.clip(p, 1e-7, 1.0)

    B, _, D, E = p.shape
    y1 = y_mup.astype(np.int64)
    y2 = y_mum.astype(np.int64)

    mup_arg = p[:, 0].argmax(axis=-1)  # (B, D)
    mum_arg = p[:, 1].argmax(axis=-1)
    top1 = 0.5 * (np.mean(mup_arg == y1) + np.mean(mum_arg == y2))
    k2   = 0.5 * (np.mean(np.abs(mup_arg - y1) <= 2) +
                  np.mean(np.abs(mum_arg - y2) <= 2))

    true1 = np.take_along_axis(p[:, 0], y1[..., None], axis=-1)[..., 0]
    true2 = np.take_along_axis(p[:, 1], y2[..., None], axis=-1)[..., 0]
    ce1 = -np.log(true1).mean()
    ce2 = -np.log(true2).mean()
    core_ce = 0.5 * (ce1 + ce2)

    return float(top1), float(k2), float(core_ce)

def evaluate_model(model, X, y_mup, y_mum, batch=64):
    preds = model.predict(X, batch_size=batch, verbose=0)
    return _metrics_from_pred(preds, y_mup, y_mum)

def run_ablation_zero_denoiser(model, X, y_mup, y_mum, batch=32):
    f_front, f_back = make_models_split(model)
    x_in = f_front.predict(X, batch_size=batch, verbose=0)  # (B,D,E,7) = [inp(1), deno(2), pos(4)]
    x_in_zero = tf.concat([x_in[..., :1],
                           tf.zeros_like(x_in[..., 1:3]),
                           x_in[..., 3:]], axis=-1)
    preds = f_back.predict(x_in_zero, batch_size=batch, verbose=0)
    return _metrics_from_pred(preds, y_mup, y_mum)

def main():
    ap = argparse.ArgumentParser(description="UNet denoiser visualization + stats + ablation")
    ap.add_argument("--model_path", type=str, required=True, help="Saved .h5/.keras model path")
    ap.add_argument("--val_root",   type=str, required=True, help="Validation ROOT file")
    ap.add_argument("--index",      type=int, default=0, help="Single sample index for plots")
    ap.add_argument("--save_dir",   type=str, default="./denoiser_vis", help="Where to save figures")
    ap.add_argument("--n_eval",     type=int, default=256, help="#samples for dataset stats/ablation")
    ap.add_argument("--batch",      type=int, default=32, help="Inference batch size")
    args = ap.parse_args()

    custom_objs = {
        "PositionalChannels": PositionalChannels,
        "ReduceMeanC": ReduceMeanC,
        "ReduceMaxC": ReduceMaxC,
        "ToSeqDet": ToSeqDet,
        "FromSeqDet": FromSeqDet,
        "L1MeanLoss": L1MeanLoss,
        "StopGradient": StopGradient,
        "ReduceMaxLast": ReduceMaxLast,
        "Custom>PositionalChannels": PositionalChannels,
        "Custom>ReduceMeanC": ReduceMeanC,
        "Custom>ReduceMaxC": ReduceMaxC,
        "Custom>ToSeqDet": ToSeqDet,
        "Custom>FromSeqDet": FromSeqDet,
        "Custom>L1MeanLoss": L1MeanLoss,
        "Custom>StopGradient": StopGradient,
        "Custom>ReduceMaxLast": ReduceMaxLast,
    }
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects=custom_objs,
        compile=False
    )
    print("[INFO] model loaded.")

    den_layer = get_denoiser_layer(model, "DenoiserUNet")
    den_out   = den_layer(model.input)                # (B,D,E,2)
    deno_model= tf.keras.Model(model.input, den_out)  # (B,D,E,2)
    print("[INFO] denoiser submodel ready.")

    X_val, y_mup_v, y_mum_v = load_data(args.val_root)
    print(f"[INFO] val samples: {len(X_val)}")

    plot_suite_readable(model, deno_model, X_val, y_mup_v, y_mum_v,
                        idx=args.index, outdir=args.save_dir, prefix=f"idx{args.index}")

    dataset_denoiser_stats(deno_model, X_val, y_mup_v, y_mum_v, n=args.n_eval, band=2)
    n = min(args.n_eval, len(X_val))
    Xb = X_val[:n]; y1b = y_mup_v[:n]; y2b = y_mum_v[:n]
    top1, k2, ce = evaluate_model(model, Xb, y1b, y2b, batch=args.batch)
    ztop1, zk2, zce = run_ablation_zero_denoiser(model, Xb, y1b, y2b, batch=args.batch)

    print("\n[Ablation] Normal vs Zero-Denoiser")
    print(f"  top1-mean-acc : {top1:.4f}  vs  {ztop1:.4f}  (Δ {top1-ztop1:+.4f})")
    print(f"  k2-acc        : {k2:.4f}    vs  {zk2:.4f}    (Δ {k2-zk2:+.4f})")
    print(f"  core CE       : {ce:.4f}    vs  {zce:.4f}    (Δ {ce-zce:+.4f})")
    print(f"[INFO] figures saved to: {os.path.abspath(args.save_dir)}")

if __name__ == "__main__":
    main()
