# Suppress warnings and logs
import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
absl.logging.set_verbosity('error')

import argparse
import ROOT
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from training_scripts import TrackFinder_prod, TrackFinder_attention
from training_scripts.TrackFinder_attention import ChannelAvgPool, ChannelMaxPool, SpatialMaxPool, SpatialAvgPool


def plot_res_histogram(model_path, y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.cast(tf.squeeze(y_muPlus_true, axis=1), tf.float32)
    y_muMinus_true = tf.cast(tf.squeeze(y_muMinus_true, axis=1), tf.float32)

    y_muPlus_pred = tf.cast(tf.argmax(tf.squeeze(y_muPlus_pred, axis=1), axis=-1), tf.float32)
    y_muMinus_pred = tf.cast(tf.argmax(tf.squeeze(y_muMinus_pred, axis=1), axis=-1), tf.float32)

    res_plus = y_muPlus_pred - y_muPlus_true        # (num_events, num_detectors)
    res_minus = y_muMinus_pred - y_muMinus_true

    res_plus = res_plus.numpy().flatten()
    res_minus = res_minus.numpy().flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(res_plus, bins=100, alpha=0.5, label='mu+')
    plt.hist(res_minus, bins=100, alpha=0.5, label='mu-')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Residual Histogram')

    plot_name = model_path.replace('models/', '').replace('.keras', '.png').replace('.h5', '.png')
    plot_path = os.path.join(os.path.dirname(__file__), "plots", plot_name)
    plt.savefig(plot_path)
    plt.show()

    mean_plus, mean_minus = np.mean(res_plus), np.mean(res_minus)
    std_plus, std_minus = np.std(res_plus), np.std(res_minus)
    return mean_plus, mean_minus, std_plus, std_minus


def evaluate_model(root_file, model_path):
    if 'track_finder.h5' in model_path:
        X, y_muPlus, y_muMinus = TrackFinder_prod.load_data(root_file)
    else:
        X, y_muPlus, y_muMinus = TrackFinder_attention.load_data(root_file)
    if X is None:
        return
    y = np.stack([y_muPlus, y_muMinus], axis=1)  # Shape: (num_events, 2, 62)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_objects = {
        "custom_loss": TrackFinder_prod.custom_loss, 
        "Adam": tf.keras.optimizers.legacy.Adam
    }
    if 'track_finder_cbam.keras' in model_path:
        custom_objects['ChannelAvgPool'] = ChannelAvgPool
        custom_objects['ChannelMaxPool'] = ChannelMaxPool
        custom_objects['SpatialAvgPool'] = SpatialAvgPool
        custom_objects['SpatialMaxPool'] = SpatialMaxPool
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)

    y_pred = model.predict(X_test)
    mean_plus, mean_minus, std_plus, std_minus = plot_res_histogram(model_path, y_test, y_pred)
    print(f'Mu+ Mean: {mean_plus:.3f}, Std Dev: {std_plus:.3f}')
    print(f'Mu- Mean: {mean_minus:.3f}, Std Dev: {std_minus:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pre-trained TrackFinder models.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.h5 or .keras).")
    args = parser.parse_args()

    print(f'Results for {args.model_path}...')
    evaluate_model(args.root_file, args.model_path)
