import sys
import uproot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from evaluate import plot_res_histogram

if len(sys.argv) != 2:
    print("Usage: python3 extract_hits.py path/to/qtracker_reco.root")
    sys.exit(1)

input_root = sys.argv[1]

f = uproot.open(input_root)
tree = f["tree"]

true_mup = tree["HitArray_mup"].array(library="np")   # shape = (N, 62)
true_mum = tree["HitArray_mum"].array(library="np")   # shape = (N, 62)
pred_mup = tree["qHitArray_mup"].array(library="np")   # shape = (N, 62)
pred_mum = tree["qHitArray_mum"].array(library="np")   # shape = (N, 62)

N = true_mup.shape[0]

y_true = np.zeros((N, 2, 62), dtype=np.float32)
y_pred = np.zeros((N, 2, 62), dtype=np.float32)

for i in range(N):
    y_true[i, 0, true_mup[i]] = 1.0   # mu+
    y_true[i, 1, true_mum[i]] = 1.0   # mu–
    y_pred[i, 0, pred_mup[i]] = 1.0
    y_pred[i, 1, pred_mum[i]] = 1.0

y_true_tf = tf.convert_to_tensor(y_true)
y_pred_tf = tf.convert_to_tensor(y_pred)

mean_plus, mean_minus, std_plus, std_minus = plot_res_histogram("models/track_finder.h5", y_true_tf, y_pred_tf)

print(f"Mu+ mean residual = {mean_plus:.3f}, σ = {std_plus:.3f}")
print(f"Mu– mean residual = {mean_minus:.3f}, σ = {std_minus:.3f}")
