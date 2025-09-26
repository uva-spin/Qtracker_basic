import os
import argparse
import absl.logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Tuple

from tensorflow.keras import layers, mixed_precision
mixed_precision.set_global_policy("mixed_float16")

try:
    from keras import ops as kops
except Exception:
    kops = None

import ROOT 
from models import data_loader 
import QTracker_prod            
from refine import refine_hit_arrays

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
        pos = tf.expand_dims(pos, 0)
        return tf.tile(pos, [B,1,1,1])                     # (B,D,E,4)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMeanC(layers.Layer):
    def call(self, x): return tf.reduce_mean(x, axis=-1, keepdims=True)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxC(layers.Layer):
    def call(self, x): return tf.reduce_max(x, axis=-1, keepdims=True)
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class ToSeqDet(layers.Layer):
    def call(self, t): return tf.reshape(tf.transpose(t, [0,2,1,3]), [-1, tf.shape(t)[1], tf.shape(t)[3]])
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class FromSeqDet(layers.Layer):
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
        if self.coef > 0.0:
            if kops is not None:
                m = kops.mean(kops.cast(x, "float32"))
            else:
                m = tf.reduce_mean(tf.cast(x, tf.float32))
            self.add_loss(self.coef * m)
        return x
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"coef": self.coef})
        return cfg
        
@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxLast(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1)
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    def get_config(self):
        return {}

@tf.keras.utils.register_keras_serializable(package="Custom")
class StopGradient(layers.Layer):
    def call(self, x): return tf.stop_gradient(x)
    def get_config(self): return {}

CUSTOM_OBJECTS: Dict[str, Any] = {
    "Custom>PositionalChannels": PositionalChannels,
    "Custom>ReduceMeanC": ReduceMeanC,
    "Custom>ReduceMaxC": ReduceMaxC,
    "Custom>ToSeqDet": ToSeqDet,
    "Custom>FromSeqDet": FromSeqDet,
    "Custom>L1MeanLoss": L1MeanLoss,
    "Custom>StopGradient": StopGradient,
    "PositionalChannels": PositionalChannels,
    "ReduceMeanC": ReduceMeanC,
    "ReduceMaxC": ReduceMaxC,
    "ToSeqDet": ToSeqDet,
    "FromSeqDet": FromSeqDet,
    "L1MeanLoss": L1MeanLoss,
    "StopGradient": StopGradient,
    "Custom>ReduceMaxLast": ReduceMaxLast,
    "ReduceMaxLast": ReduceMaxLast,
}

def plot_residuals(det_ids, res_plus, res_minus, model_path, stage_label):
    mean_p  = np.nanmean(np.abs(res_plus), axis=0)
    std_p   = np.nanstd(np.abs(res_plus), axis=0)
    mean_m  = np.nanmean(np.abs(res_minus), axis=0)
    std_m   = np.nanstd(np.abs(res_minus), axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(det_ids, mean_p, yerr=std_p, marker='o', label='μ+ mean±σ')
    plt.errorbar(det_ids, mean_m, yerr=std_m, marker='s', label='μ- mean±σ')
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Detector Layer (skipping masked slots)')
    plt.ylabel('Absolute Residual (predicted − true)')
    plt.title(f'Per-layer Absolute Residual ({stage_label.capitalize()})')
    plt.legend()
    plt.tight_layout()

    base     = os.path.splitext(os.path.basename(model_path))[0]
    fname    = f"{base}_{stage_label}_residuals.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()

def _import_builders() -> Dict[str, Callable[..., tf.keras.Model]]:
    builders: Dict[str, Callable[..., tf.keras.Model]] = {}
    try:
        from models import TrackFinder_unetpp_ds_v5 as _v5
        builders["TrackFinder_unetpp_ds"] = _v5.build_model
    except Exception:
        pass
    try:
        from models import TrackFinder_unetpp_ds as _base
        if "TrackFinder_unetpp_ds" not in builders:
            builders["TrackFinder_unetpp_ds"] = _base.build_model
        else:
            builders["TrackFinder_unetpp_ds_base"] = _base.build_model
    except Exception:
        pass

    if not builders:
        raise RuntimeError("No available TrackFinder builders found in models package.")
    return builders


def build_from_weights(weights_path: str, model_key: str, use_bn: bool, base: int) -> tf.keras.Model:
    builders = _import_builders()
    if model_key not in builders:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(builders.keys())}")
    build_model = builders[model_key]
    model = build_model(num_detectors=62, num_elementIDs=201, use_bn=use_bn, base=base)
    model.load_weights(weights_path)
    return model


def load_model_file(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )

def _extract_pred_2DDE(y_pred: Any) -> np.ndarray:
    if isinstance(y_pred, dict):
        if "pred" in y_pred:
            y = y_pred["pred"]
        else:
            first_key = list(y_pred.keys())[0]
            y = y_pred[first_key]
    else:
        y = y_pred
    if tf.is_tensor(y):
        y = y.numpy()
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    if y.ndim == 4 and y.shape[1] == 2:
        return y
    if y.ndim == 4 and y.shape[-1] == 2:
        return np.transpose(y, (0, 3, 1, 2))
    raise ValueError(f"Unexpected prediction shape: {y.shape}. Expect (B,2,D,E) or (B,D,E,2).")

def evaluate_model(args):
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(args.root_file)
    if X_test is None:
        print("[ERROR] Failed to load data from ROOT file.")
        return

    y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=1)  # (B,2,D)
    det_test, elem_test, _, _, _ = QTracker_prod.load_detector_element_data(args.root_file)

    mask = np.ones(62, dtype=bool)
    mask[6:12]   = False  # 7-12
    mask[54:62]  = False  # 55-62

    if args.model_path.endswith(".weights.h5"):
        if not args.model:
            raise ValueError("--model.")
        model = build_from_weights(args.model_path,
                                   model_key=args.model,
                                   use_bn=bool(args.batch_norm),
                                   base=int(args.base))
        y_pred = model.predict(X_test, verbose=0)
    else:
        model = load_model_file(args.model_path)
        y_pred = model.predict(X_test, verbose=0)

    y_pred_2dde = _extract_pred_2DDE(y_pred)

    y_p_raw = np.argmax(y_pred_2dde[:, 0, :, :], axis=-1).astype(np.int32)  # (B,D)
    y_m_raw = np.argmax(y_pred_2dde[:, 1, :, :], axis=-1).astype(np.int32)  # (B,D)

    y_p_true = y_test[:, 0, :].astype(np.int32)
    y_m_true = y_test[:, 1, :].astype(np.int32)

    raw_p_res = y_p_raw - y_p_true
    raw_m_res = y_m_raw - y_m_true

    print("\n--- Raw Residuals (Before Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(raw_p_res[:, det])), np.std(np.abs(raw_p_res[:, det]))
        m_m, s_m = np.mean(np.abs(raw_m_res[:, det])), np.std(np.abs(raw_m_res[:, det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = (np.where(mask)[0] + 1)
    plot_residuals(dets_used, raw_p_res[:, mask], raw_m_res[:, mask], args.model_path, "raw")

    ref_p, ref_m = refine_hit_arrays(y_p_raw, y_m_raw, det_test, elem_test)
    ref_p_res = ref_p - y_p_true
    ref_m_res = ref_m - y_m_true

    print("\n--- Refined Residuals (After Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(ref_p_res[:, det])), np.std(np.abs(ref_p_res[:, det]))
        m_m, s_m = np.mean(np.abs(ref_m_res[:, det])), np.std(np.abs(ref_m_res[:, det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = (np.where(mask)[0] + 1)
    plot_residuals(dets_used, ref_p_res[:, mask], ref_m_res[:, mask], args.model_path, "refined")

    acc_p = np.mean(np.abs(raw_p_res) == 0)
    acc_m = np.mean(np.abs(raw_m_res) == 0)
    print(f"\nRaw μ+ accuracy: {acc_p:.4f}")
    print(f"Raw μ- accuracy: {acc_m:.4f}")

    acc_p2 = np.mean(np.abs(raw_p_res) <= 2)
    acc_m2 = np.mean(np.abs(raw_m_res) <= 2)
    print(f"Raw μ+ within-2 accuracy: {acc_p2:.4f}")
    print(f"Raw μ- within-2 accuracy: {acc_m2:.4f}")

    racc_p = np.mean(np.abs(ref_p_res) == 0)
    racc_m = np.mean(np.abs(ref_m_res) == 0)
    print(f"\nRefined μ+ accuracy: {racc_p:.4f}")
    print(f"Refined μ- accuracy: {racc_m:.4f}")

    racc_p2 = np.mean(np.abs(ref_p_res) <= 2)
    racc_m2 = np.mean(np.abs(ref_m_res) <= 2)
    print(f"Refined μ+ within-2 accuracy: {racc_p2:.4f}")
    print(f"Refined μ- within-2 accuracy: {racc_m2:.4f}")

    print("\n--- Raw Absolute Residuals (Before Refinement) ---")
    print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    m_p, s_p = np.mean(np.abs(raw_p_res)),  np.std(np.abs(raw_p_res))
    m_m, s_m = np.mean(np.abs(raw_m_res)),  np.std(np.abs(raw_m_res))
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    print("\n--- Refined Absolute Residuals (After Refinement) ---")
    print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    m_p, s_p = np.mean(np.abs(ref_p_res)),  np.std(np.abs(ref_p_res))
    m_m, s_m = np.mean(np.abs(ref_m_res)),  np.std(np.abs(ref_m_res))
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained TrackFinder models.")
    parser.add_argument("root_file",  type=str, help="Path to the val/test ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model (.h5/.keras) or weights (.weights.h5).")
    parser.add_argument("--batch_norm", type=int, default=1, help="[0|1] for BN in code-built models")
    parser.add_argument("--base",       type=int, default=64, help="Base channels for code-built models")
    parser.add_argument("--model",      type=str, default="TrackFinder_unetpp_ds", help="Model key when using .weights.h5")
    args = parser.parse_args()

    print(f"\nResults for {args.model_path}...")
    evaluate_model(args)

if __name__ == "__main__":
    main()
