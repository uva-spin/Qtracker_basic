import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
absl.logging.set_verbosity('error')

import argparse
import ROOT
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable, Dict

# core TrackFinder loaders / custom loss
from models import data_loader
from models import (
    TrackFinder_unetpp_ds,
    # TrackFinder_unet_3p_ds,
)
import QTracker_prod
from refine import refine_hit_arrays

def plot_residuals(det_ids, res_plus, res_minus, model_path, stage_label):
    mean_p  = np.nanmean(np.abs(res_plus), axis=0)
    std_p   = np.nanstd(np.abs(res_plus), axis=0)
    mean_m  = np.nanmean(np.abs(res_minus), axis=0)
    std_m   = np.nanstd(np.abs(res_minus), axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(det_ids, mean_p, yerr=std_p, marker='o', label='μ+ mean±σ')
    plt.errorbar(det_ids, mean_m, yerr=std_m, marker='s', label='μ- mean±σ')
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel(f'Detector Layer (skipping masked slots)')
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

def evaluate_model(args):
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(args.root_file)
    if X_test is None:
        return

    y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=1)
    det_test, elem_test, _, _, _ = QTracker_prod.load_detector_element_data(args.root_file)

    mask = np.ones(62, dtype=bool)
    mask[6:12]   = False
    mask[54:62]  = False

    denoise_model = tf.keras.models.load_model(
        args.denoise_model_path, 
        compile=False,
    )
    seg_model = tf.keras.models.load_model(
        args.seg_model_path, 
        compile=False,
    )

    X_sigmoid = denoise_model.predict(X_test)
    X_clean = (X_sigmoid > 0.5).astype(np.float32)
    y_pred  = seg_model.predict(X_clean)

    y_p_raw = tf.cast(
        tf.argmax(tf.squeeze(tf.split(y_pred,2,axis=1)[0],axis=1), axis=-1),
        tf.int32
    ).numpy()
    y_m_raw = tf.cast(
        tf.argmax(tf.squeeze(tf.split(y_pred,2,axis=1)[1],axis=1), axis=-1),
        tf.int32
    ).numpy()

    y_p_true = y_test[:,0,:].astype(np.int32)
    y_m_true = y_test[:,1,:].astype(np.int32)

    raw_p_res = y_p_raw - y_p_true
    raw_m_res = y_m_raw - y_m_true

    print("\n--- Raw Residuals (Before Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(raw_p_res[:,det])), np.std(np.abs(raw_p_res[:,det]))
        m_m, s_m = np.mean(np.abs(raw_m_res[:,det])), np.std(np.abs(raw_m_res[:,det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = (np.where(mask)[0] + 1)
    plot_residuals(dets_used, raw_p_res[:,mask], raw_m_res[:,mask], args.seg_model_path, 'raw')

    ref_p, ref_m = refine_hit_arrays(
        y_p_raw, y_m_raw, det_test, elem_test
    )
    ref_p_res = ref_p - y_p_true
    ref_m_res = ref_m - y_m_true

    print("\n--- Refined Residuals (After Refinement, all events) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(ref_p_res[:,det])),  np.std(np.abs(ref_p_res[:,det]))
        m_m, s_m = np.mean(np.abs(ref_m_res[:,det])),  np.std(np.abs(ref_m_res[:,det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    dets_used = (np.where(mask)[0] + 1)
    plot_residuals(dets_used, ref_p_res[:,mask], ref_m_res[:,mask], args.seg_model_path, 'refined')

    acc_p = np.mean(np.abs(raw_p_res) == 0)
    acc_m = np.mean(np.abs(raw_m_res) == 0)
    print(f"\nRaw μ+ accuracy: {acc_p:.4f}")
    print(f"Raw μ- accuracy: {acc_m:.4f}")

    acc_p = np.mean(np.abs(raw_p_res) <= 2)
    acc_m = np.mean(np.abs(raw_m_res) <= 2)
    print(f"\nRaw μ+ within-2 accuracy: {acc_p:.4f}")
    print(f"Raw μ- within-2 accuracy: {acc_m:.4f}")

    acc_p = np.mean(np.abs(ref_p_res) == 0)
    acc_m = np.mean(np.abs(ref_m_res) == 0)
    print(f"\nRefined μ+ accuracy: {acc_p:.4f}")
    print(f"Refined μ- accuracy: {acc_m:.4f}")

    acc_p = np.mean(np.abs(ref_p_res) <= 2)
    acc_m = np.mean(np.abs(ref_m_res) <= 2)
    print(f"\nRefined μ+ within-2 accuracy: {acc_p:.4f}")
    print(f"Refined μ- within-2 accuracy: {acc_m:.4f}")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models."
    )
    parser.add_argument("root_file",  type=str, help="Path to the val/test ROOT file.")
    parser.add_argument("denoise_model_path", type=str, help="Path to the saved denoiser model file (.h5 or .keras).")
    parser.add_argument("seg_model_path", type=str, help="Path to the saved segmentation model file (.h5 or .keras).")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--base", type=int, default=64, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    args = parser.parse_args()

    print(f"\nEvaluating...")
    evaluate_model(args)
