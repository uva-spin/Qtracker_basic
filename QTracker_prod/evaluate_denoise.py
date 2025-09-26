import argparse
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from models import data_loader
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Denoise Model")
    parser.add_argument("--val_file_path", type=str, default="data/processed_files/mc_events_val_denoise.root")
    parser.add_argument("--denoise_model_path", type=str, default="checkpoints/track_finder_denoise.h5")
    args = parser.parse_args()

    X_test, X_clean_test, _, _ = data_loader.load_data_denoise(
        args.val_file_path
    )

    denoise_model = tf.keras.models.load_model(
        args.denoise_model_path,
        compile=False,
    )

    X_clean_pred = denoise_model.predict(X_test)
    X_clean_pred = (X_clean_pred > 0.5).astype(np.float32)

    cm = confusion_matrix(X_clean_test.flatten(), X_clean_pred.flatten())

    print("Results (Confusion Matrix + Accuracy):")
    print(classification_report(X_clean_test.flatten(), X_clean_pred.flatten(), digits=4))


    # Plot heatmap
    X_test = X_test[0, :, :, 0].T[::-1]
    X_clean_pred = X_clean_pred[0, :, :, 0].T[::-1]
    X_clean_test = X_clean_test[0, :, :, 0].T[::-1]

    plt.imshow(X_test, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("2D Array Heatmap")
    plt.savefig("plots/original.png", dpi=300, bbox_inches='tight')
    plt.close()  # optional: closes the figure so it doesn’t stay in memory

    plt.imshow(X_clean_pred, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("2D Array Heatmap")
    plt.savefig("plots/denoised.png", dpi=300, bbox_inches='tight')
    plt.close()  # optional: closes the figure so it doesn’t stay in memory

    plt.imshow(X_clean_test, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("2D Array Heatmap")
    plt.savefig("plots/clean.png", dpi=300, bbox_inches='tight')
    plt.close()  # optional: closes the figure so it doesn’t stay in memory
