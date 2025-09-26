import ROOT
import numpy as np
import matplotlib.pyplot as plt
from models import data_loader
import tensorflow as tf

X_test, y_muPlus_test, y_muMinus_test = data_loader.load_data(
    "data/processed_files/mc_events_val_denoise.root"
)

denoise_model = tf.keras.models.load_model(
    "checkpoints/track_finder_denoise.h5", 
    compile=False,
)

X_clean = denoise_model.predict(X_test)
X_clean = (X_clean > 0.5).astype(np.float32)

X_test = X_test[0, :, :, 0].T[::-1]
X_clean = X_clean[0, :, :, 0].T[::-1]

plt.imshow(X_test, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Value")
plt.title("2D Array Heatmap")
plt.savefig("original.png", dpi=300, bbox_inches='tight')
plt.close()  # optional: closes the figure so it doesn’t stay in memory

plt.imshow(X_clean, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Value")
plt.title("2D Array Heatmap")
plt.savefig("denoised.png", dpi=300, bbox_inches='tight')
plt.close()  # optional: closes the figure so it doesn’t stay in memory
