# evaluate.py

# -----------------------------------------------------------------------------
# Suppress TensorFlow/absl logging
import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # only show ERROR
absl.logging.set_verbosity('error')

import argparse
import ROOT
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# core TrackFinder loaders / custom loss
from training_scripts import TrackFinder_prod, TrackFinder_attention
from training_scripts.TrackFinder_attention import (
    ChannelAvgPool, ChannelMaxPool, SpatialAvgPool, SpatialMaxPool
)

# import QTracker_prod so we can load true detector/element data and refinement
import QTracker_prod
# -----------------------------------------------------------------------------

def plot_residuals(det_ids, res_plus, res_minus, model_path, stage_label):
    """
    Error‐bar plot of residuals (mean ± std) for μ⁺ and μ⁻ vs. detector ID.
    Only uses the provided det_ids / sliced res arrays.
    """
    mean_p  = np.mean(np.abs(res_plus), axis=0)
    std_p   = np.std(np.abs(res_plus), axis=0)
    mean_m  = np.mean(np.abs(res_minus), axis=0)
    std_m   = np.std(np.abs(res_minus), axis=0)

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
    # fname    = f"{stage_label}_residuals.png"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, fname))
    plt.show()


def evaluate_model(root_file, model_path):
    # 1) load X, y
    if 'track_finder.h5' in model_path:
        X, y_muPlus, y_muMinus = TrackFinder_prod.load_data(root_file)
    else:
        X, y_muPlus, y_muMinus = TrackFinder_attention.load_data(root_file)
    if X is None:
        return

    # 2) stack ground truth, grab detector/element arrays
    y = np.stack([y_muPlus, y_muMinus], axis=1)  # (n_events, 2, num_detectors)
    detectorIDs, elementIDs, _, _, _ = QTracker_prod.load_detector_element_data(root_file)
    N = QTracker_prod.NUM_DETECTORS

    # build a boolean mask of “used” detectors
    mask = np.ones(N, dtype=bool)
    mask[7:12]  = False  # unused station-1
    mask[55:58] = False  # DP-1
    mask[59:62] = False  # DP-2

    # 3) split everything with the same seed
    (X_train, X_test,
     y_train, y_test,
     det_train, det_test,
     elem_train, elem_test) = train_test_split(
        X, y, detectorIDs, elementIDs,
        test_size=0.2, random_state=42
    )

    # 4) load model with custom objects
    custom_objects = {
        "custom_loss": TrackFinder_prod.custom_loss,
        "Adam":        tf.keras.optimizers.legacy.Adam
    }
    if 'track_finder_cbam.keras' in model_path:
        custom_objects.update({
            'ChannelAvgPool': ChannelAvgPool,
            'ChannelMaxPool': ChannelMaxPool,
            'SpatialAvgPool': SpatialAvgPool,
            'SpatialMaxPool': SpatialMaxPool
        })
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    # 5) predict on test set
    y_pred = model.predict(X_test)

    # 6) compute raw residuals
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

    raw_p_res = y_p_raw  - y_p_true
    raw_m_res = y_m_raw  - y_m_true

    # 7) print per-layer stats BEFORE refinement (only unmasked)
    print("\n--- Raw Absolute Residuals (Before Refinement) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(raw_p_res[:,det])),  np.std(np.abs(raw_p_res[:,det]))
        m_m, s_m = np.mean(np.abs(raw_m_res[:,det])),  np.std(np.abs(raw_m_res[:,det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    # 8) plot raw residuals
    dets_used = (np.where(mask)[0] + 1)  # 1-based IDs for display
    plot_residuals(dets_used, raw_p_res[:,mask], raw_m_res[:,mask], model_path, 'raw')

    # 9) refinement
    ref_p, ref_m = QTracker_prod.refine_hit_arrays(
        y_p_raw, y_m_raw, det_test, elem_test
    )
    ref_p_res = ref_p - y_p_true
    ref_m_res = ref_m - y_m_true

    # 10) print per-layer stats AFTER refinement (only unmasked)
    print("\n--- Refined Absolute Residuals (After Refinement) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = np.mean(np.abs(ref_p_res[:,det])),  np.std(np.abs(ref_p_res[:,det]))
        m_m, s_m = np.mean(np.abs(ref_m_res[:,det])),  np.std(np.abs(ref_m_res[:,det]))
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    # 11) plot refined residuals
    plot_residuals(dets_used, ref_p_res[:,mask], ref_m_res[:,mask], model_path, 'refined')

    # 12) Print summary of results
    print("\n--- Raw Absolute Residuals (Before Refinement) ---")
    print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    m_p, s_p = np.mean(np.abs(raw_p_res)),  np.std(np.abs(raw_p_res))
    m_m, s_m = np.mean(np.abs(raw_m_res)),  np.std(np.abs(raw_m_res))
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    print("\n--- Refined Absolute Residuals (Before Refinement) ---")
    print("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    m_p, s_p = np.mean(np.abs(ref_p_res)),  np.std(np.abs(ref_p_res))
    m_m, s_m = np.mean(np.abs(ref_m_res)),  np.std(np.abs(ref_m_res))
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models."
    )
    parser.add_argument("root_file",  type=str, help="Path to the combined ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.h5 or .keras).")
    args = parser.parse_args()

    print(f"\nResults for {args.model_path}...")
    evaluate_model(args.root_file, args.model_path)
