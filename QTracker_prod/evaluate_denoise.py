import argparse
import ROOT
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from models import data_loader
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Denoise Model")
    parser.add_argument("--val_file_path", type=str, default="data/processed_files/mc_events_val.root")
    parser.add_argument("--model_path", type=str, default="checkpoints/track_finder_denoise.h5")
    args = parser.parse_args()

    X_test, X_clean_test, _, _ = data_loader.load_data_denoise(
        args.val_file_path
    )

    denoise_model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
    )

    X_clean_pred = denoise_model.predict(X_test)

    print("Results (Confusion Matrix + Accuracy):")
    print(classification_report(
        X_clean_test.flatten(), 
        (X_clean_pred > 0.5).astype(np.float32).flatten(), 
        digits=4
    ))
    print(classification_report(
        X_clean_test.flatten(), 
        (X_clean_pred > 0.4).astype(np.float32).flatten(), 
        digits=4
    ))
    print(classification_report(
        X_clean_test.flatten(), 
        (X_clean_pred > 0.3).astype(np.float32).flatten(), 
        digits=4
    ))
    print(classification_report(
        X_clean_test.flatten(), 
        X_test.flatten(), 
        digits=4
    ))

    # Plot heatmap
    X_test = (X_test[1, :, :, 0].T[::-1] > 0.5).astype(np.int32)
    X_clean_pred = ((X_clean_pred > 0.5).astype(np.int32)[1, :, :, 0].T[::-1])
    X_clean_test = (X_clean_test[1, :, :, 0].T[::-1]).astype(np.int32)

    def make_error_map(true_mask, pred_mask):
        """
        Create categorical error map:
        0 = TN, 1 = FN, 2 = TP, 3 = FP
        """
        error_map = np.zeros_like(true_mask, dtype=np.int32)
        error_map[(true_mask == 1) & (pred_mask == 0)] = 1  # FN
        error_map[(true_mask == 1) & (pred_mask == 1)] = 2  # TP
        error_map[(true_mask == 0) & (pred_mask == 1)] = 3  # FP
        return error_map

    # Define colormap (categorical)
    cmap = mcolors.ListedColormap(["black", "red", "green", "blue"])
    bounds = [0,1,2,3,4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- 1. Original Input compared to Ground Truth ---
    error_map_original = make_error_map(X_clean_test, X_test)

    plt.figure(figsize=(6,6))
    im = plt.imshow(error_map_original, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ticks=[0.5,1.5,2.5,3.5])
    cbar.ax.set_yticklabels(["TN","FN","TP","FP"])
    plt.title("Original vs Ground Truth (Error Map)")
    plt.savefig("plots/original.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2. Denoised Prediction compared to Ground Truth ---
    error_map_denoised = make_error_map(X_clean_test, X_clean_pred)

    plt.figure(figsize=(6,6))
    im = plt.imshow(error_map_denoised, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ticks=[0.5,1.5,2.5,3.5])
    cbar.ax.set_yticklabels(["TN","FN","TP","FP"])
    plt.title("Denoised vs Ground Truth (Error Map)")
    plt.savefig("plots/denoised.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.imshow(X_clean_pred, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("2D Array Heatmap")
    plt.savefig("plots/denoised_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()  # optional: closes the figure so it doesn’t stay in memory

    plt.imshow(X_clean_test, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("2D Array Heatmap")
    plt.savefig("plots/clean.png", dpi=300, bbox_inches='tight')
    plt.close()  # optional: closes the figure so it doesn’t stay in memory
