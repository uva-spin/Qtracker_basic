"""
Saves a validation confusion-matrix PNG every `freq` epochs, as frames meant
to be stitched into a GIF (see make_gif.py) -- watching the confusion matrix
across an entire curriculum run is a direct visual of the catastrophic
forgetting seen at the low/med -> high phase transition: rows should sharpen
toward the diagonal, then blur again right when the high phase starts.
"""

import os

import numpy as np
import tensorflow as tf

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """Predicts on X_val every `freq` epochs and saves a labeled confusion-matrix PNG."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str,
        num_classes: int,
        freq: int = 2,
        batch_size: int = 256,
    ):
        super().__init__()
        self._X_val = X_val
        self._y_val = y_val
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._freq = freq
        self._batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not MATPLOTLIB_AVAILABLE or (epoch + 1) % self._freq != 0:
            return

        preds = self.model.predict(self._X_val, batch_size=self._batch_size, verbose=0)
        pred_classes = np.argmax(preds, axis=-1)
        cm = _confusion_matrix(self._y_val, pred_classes, self._num_classes)
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm.astype(np.float64) / row_sums

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
        ax.set_xlabel("Predicted pair count")
        ax.set_ylabel("True pair count")
        ax.set_title(f"Epoch {epoch + 1}")
        ax.set_xticks(range(self._num_classes))
        ax.set_yticks(range(self._num_classes))
        fig.colorbar(im, ax=ax, label="Row-normalized fraction")
        fig.tight_layout()
        fig.savefig(os.path.join(self._output_dir, f"epoch_{epoch + 1:03d}.png"))
        plt.close(fig)
