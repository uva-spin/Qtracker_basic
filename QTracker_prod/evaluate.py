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

    res_plus = (y_muPlus_pred - y_muPlus_true)       # (num_events, num_detectors)
    res_minus = (y_muMinus_pred - y_muMinus_true)

    res_plus = res_plus.numpy()
    res_minus = res_minus.numpy()

    true_plus = y_muPlus_true.numpy()   
    true_minus = y_muMinus_true.numpy() 

    all_res_plus = res_plus.flatten()
    all_res_minus = res_minus.flatten()

    num_layers = res_plus.shape[1]
    per_slot_stats = []
    for layer_idx in range(num_layers):
        plus_residuals  = res_plus[:, layer_idx]
        minus_residuals = res_minus[:, layer_idx]

        mean_plus  = np.mean(plus_residuals)
        std_plus   = np.std(plus_residuals)
        mean_minus = np.mean(minus_residuals)
        std_minus  = np.std(minus_residuals)

        per_slot_stats.append((
            layer_idx+1,
            mean_plus, std_plus,
            mean_minus, std_minus
        ))

    per_slot_stats_matched = []  
    for layer_idx in range(num_layers):
        mask_plus  = true_plus[:, layer_idx] > 0   
        mask_minus = true_minus[:, layer_idx] > 0  

        if np.any(mask_plus):  
            plus_residuals_matched = res_plus[mask_plus, layer_idx]  
            mean_plus_m  = np.mean(plus_residuals_matched)           
            std_plus_m   = np.std(plus_residuals_matched)            
        else:
            mean_plus_m, std_plus_m = 0.0, 0.0                        

        if np.any(mask_minus): 
            minus_residuals_matched = res_minus[mask_minus, layer_idx]  
            mean_minus_m = np.mean(minus_residuals_matched)             
            std_minus_m  = np.std(minus_residuals_matched)              
        else:
            mean_minus_m, std_minus_m = 0.0, 0.0                         

        per_slot_stats_matched.append((  
            layer_idx+1,                                           
            mean_plus_m, std_plus_m, mean_minus_m, std_minus_m     
        ))                                                         

    print("\nPer-slot residual statistics (all detectors):")
    print(" layer |   μ⁺ mean   |  μ⁺ σ   |  μ- mean   |  μ- σ")
    print("--------------------------------------------------")
    for (layer, m_p, s_p, m_m, s_m) in per_slot_stats:
        print(f" {layer:2d}   |   {m_p:+6.3f}  | {s_p:6.3f} |   {m_m:+6.3f}  | {s_m:6.3f}")

    print("\nPer-slot residual statistics (matched only):")  
    print(" layer |   μ⁺ mean   |  μ⁺ σ   |  μ- mean   |  μ- σ")  
    print("--------------------------------------------------")  
    for (layer, m_p_m, s_p_m, m_m_m, s_m_m) in per_slot_stats_matched:  
        print(f" {layer:2d}   |   {m_p_m:+6.3f}  | {s_p_m:6.3f} |   {m_m_m:+6.3f}  | {s_m_m:6.3f}")  

    layers = np.arange(1, num_layers+1)

    mu_plus_means  = np.array([x[1] for x in per_slot_stats])
    mu_minus_means = np.array([x[3] for x in per_slot_stats])


    mu_plus_means_m  = np.array([x[1] for x in per_slot_stats_matched])  
    mu_minus_means_m = np.array([x[3] for x in per_slot_stats_matched])  

    width = 0.4
    plt.figure(figsize=(12, 5))
    plt.bar(layers - width/2, mu_plus_means, width, label='μ⁺ mean (all)') 
    plt.bar(layers + width/2, mu_minus_means, width, label='μ- mean (all)') 
    plt.xlabel("Detector Layer (1 … 62)")
    plt.ylabel("Residual (predicted − true)")
    plt.title("Per-layer Residual Means (All Detectors)")
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "per_layer_means_all.png"))
    plt.show()
 
    plt.figure(figsize=(12, 5))  
    plt.bar(layers - width/2, mu_plus_means_m, width, label='μ⁺ mean (matched)')   
    plt.bar(layers + width/2, mu_minus_means_m, width, label='μ- mean (matched)') 
    plt.xlabel("Detector Layer (1 … 62)")  
    plt.ylabel("Residual (predicted − true)")  
    plt.title("Per-layer Residual Means (Matched Only)")  
    plt.axhline(0, color='black', lw=1, linestyle='--')  
    plt.legend()  
    plt.tight_layout()  
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "per_layer_means_matched.png"))  
    plt.show()  

    global_mean_plus  = np.mean(all_res_plus)
    global_std_plus   = np.std(all_res_plus)
    global_mean_minus = np.mean(all_res_minus)
    global_std_minus  = np.std(all_res_minus)
    return (global_mean_plus, global_std_plus, global_mean_minus, global_std_minus)


def evaluate_model(root_file, model_path):
    if 'track_finder.h5' in model_path:
        X, y_muPlus, y_muMinus = TrackFinder_prod.load_data(root_file)
    else:
        X, y_muPlus, y_muMinus = TrackFinder_attention.load_data(root_file)
    if X is None:
        return

    y = np.stack([y_muPlus, y_muMinus], axis=1)  
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
    gmp, gsp, gmm, gsm = plot_res_histogram(model_path, y_test, y_pred)
    print(f"\nGlobal μ⁺: mean={gmp:.3f}, σ={gsp:.3f}")
    print(f"Global μ-: mean={gmm:.3f}, σ={gsm:.3f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pre-trained TrackFinder models.")
    parser.add_argument("root_file", type=str, help="Path to the combined ROOT file.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.h5 or .keras).")
    args = parser.parse_args()

    print(f'Results for {args.model_path}...')
    evaluate_model(args.root_file, args.model_path)
