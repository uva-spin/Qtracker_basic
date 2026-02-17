import numpy as np

from data_loader import load_data_denoise

NUM_DETECTORS = 62
MAX_PAIRS = 5

# ---- Load exactly like your training code ----
X, X_clean, y_muPlus, y_muMinus = load_data_denoise(
    "data/multi_track/processed_files/mc_events_train_low.root",
    multi_track=True,
    max_pairs=MAX_PAIRS,
)

print("\nRaw shapes from loader:")
print("y_muPlus:", y_muPlus.shape)
print("y_muMinus:", y_muMinus.shape)

# Stack exactly like your training code
y = np.stack([y_muPlus, y_muMinus], axis=2)

print("\nStacked shape:")
print("y:", y.shape)
