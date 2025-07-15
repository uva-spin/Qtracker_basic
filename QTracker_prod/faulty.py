# evaluate.py

# -----------------------------------------------------------------------------
# Suppress TensorFlow/absl logging
import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # only show ERROR
absl.logging.set_verbosity('error')

import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# core TrackFinder loaders / custom loss
from training_scripts import TrackFinder_prod, TrackFinder_attention
from training_scripts.TrackFinder_attention import (
    ChannelAvgPool, ChannelMaxPool, SpatialAvgPool, SpatialMaxPool
)

# import QTracker_prod so we can load true detector/element data and refinement
import QTracker_prod
# -----------------------------------------------------------------------------

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

    # # 7) print per-layer stats BEFORE refinement (only unmasked)
    # print("\n--- Raw Residuals (Before Refinement) ---")
    # print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    # for det in np.where(mask)[0]:
    #     m_p, s_p = raw_p_res[:,det].mean(),  raw_p_res[:,det].std()
    #     m_m, s_m = raw_m_res[:,det].mean(),  raw_m_res[:,det].std()
    #     print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")

    # 9) refinement
    y_p_ref, y_m_ref = QTracker_prod.refine_hit_arrays(
        y_p_raw, y_m_raw, det_test, elem_test
    )
    ref_p_res = y_p_ref - y_p_true
    ref_m_res = y_m_ref - y_m_true

    print("\n--- Unique Values ---")
    print("--- True Hit Array ---")
    print(np.unique(y_p_true))

    print("\n--- Raw Hit Array ---")
    print(np.unique(y_p_raw))

    print("\n--- POSITIVE ---")
    print("--- Raw Hit (Before Refinement) ---")
    print(y_p_raw[40:50, 0:2])

    print("\n--- True Hit (Before Refinement) ---")
    print(y_p_true[40:50, 0:2])

    print("\n--- Refined Hit (After Refinement) ---")
    print(y_p_ref[40:50, 0:2])

    print("\n\n--- NEGATIVE ---")
    print("--- Raw Hit (Before Refinement) ---")
    print(y_m_raw[40:50, 0:2])

    print("\n--- True Hit (Before Refinement) ---")
    print(y_m_true[40:50, 0:2])

    print("\n--- Refined Hit (After Refinement) ---")
    print(y_m_ref[40:50, 0:2])

    print("\n--- Element IDs ---")
    print([elem_test[16][i] for i, _ in enumerate(elem_test[16]) if det_test[16][i] == 3])

    # 10) print per-layer stats AFTER refinement (only unmasked)
    print("\n--- Refined Residuals (After Refinement) ---")
    print("Det |  μ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    for det in np.where(mask)[0]:
        m_p, s_p = ref_p_res[20:30,det].mean(),  ref_p_res[20:30,det].std()
        m_m, s_m = ref_m_res[20:30,det].mean(),  ref_m_res[20:30,det].std()
        print(f"{det+1:3d} | {m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")
    
    print("\nμ+ mean  |  μ+ std   |  μ- mean  |  μ- std")
    m_p, s_p = ref_p_res.mean(),  ref_p_res.std()
    m_m, s_m = ref_m_res.mean(),  ref_m_res.std()
    print(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models."
    )
    parser.add_argument("root_file",  type=str, help="Path to the combined ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.h5 or .keras).")
    args = parser.parse_args()

    evaluate_model(args.root_file, args.model_path)
