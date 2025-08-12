""" Evaluate function of track count (occupancy) vs. accuracy/precision """

import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
absl.logging.set_verbosity('error')

import argparse
import ROOT
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# core TrackFinder loaders / custom loss
from models import data_loader
from models.losses import custom_loss
import QTracker_prod
from refine import refine_hit_arrays


def evaluate_model(root_file, model_path, output):
    X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(root_file)
    if X_test is None:
        return

    y_test = np.stack([y_muPlus_test, y_muMinus_test], axis=1)
    det_test, elem_test, _, _, _ = QTracker_prod.load_detector_element_data(root_file)
    N = QTracker_prod.NUM_DETECTORS

    custom_objects = {
        "custom_loss": custom_loss,
        "Adam":        tf.keras.optimizers.legacy.Adam
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    y_pred = model.predict(X_test)  # shape: (num_events, 2, num_detectors, num_elementids)

    softmax_mup = y_pred[:, 0, :, :]   # (num_events, num_detectors, num_elementids)
    softmax_mum = y_pred[:, 1, :, :]   # (num_events, num_detectors, num_elementids)

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

    acc_p = np.mean(np.abs(raw_p_res) == 0)
    acc_m = np.mean(np.abs(raw_m_res) == 0)

    acc_p_2 = np.mean(np.abs(raw_p_res) <= 2)
    acc_m_2 = np.mean(np.abs(raw_m_res) <= 2)

    m_p, s_p = np.mean(np.abs(raw_p_res)),  np.std(np.abs(raw_p_res))
    m_m, s_m = np.mean(np.abs(raw_m_res)),  np.std(np.abs(raw_m_res))

    with open(output, "a") as file:
        file.write(f"Raw μ+ accuracy: {acc_p:.4f}\n")
        file.write(f"Raw μ- accuracy: {acc_m:.4f}\n")
        file.write(f"Raw μ+ distance-based accuracy: {acc_p_2:.4f}\n")
        file.write(f"Raw μ- distance-based accuracy: {acc_m_2:.4f}\n")
        file.write("μ+ mean  |  μ+ std   |  μ- mean  |  μ- std\n")
        file.write(f"{m_p:8.3f} | {s_p:8.3f} | {m_m:8.3f} | {s_m:8.3f}\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained TrackFinder models."
    )
    parser.add_argument("root_file",  type=str, help="Path to the val/test ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.h5 or .keras).")
    parser.add_argument("--output", type=str, default="results/func.txt", help="Path to the output result file.")
    args = parser.parse_args()

    evaluate_model(args.root_file, args.model_path, args.output)
