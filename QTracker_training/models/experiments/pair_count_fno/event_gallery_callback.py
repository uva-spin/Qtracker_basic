"""
Saves a hit-matrix heatmap for one tracked validation event, labeled with
true vs. predicted pair count, every `freq` epochs -- frames for make_gif.py.

Follows plot_HitMatrix_color.py's conventions (QTracker_prod/Util/): origin
"lower", detector ID on the x-axis, element ID on the y-axis, plain light
background. Unlike that script (which reads HitArray_mup/HitArray_mum
straight from a ROOT file and colors each cell by which signal track fired
it), this plots the classifier's actual noisy input X_val -- signal and
background are already merged into one occupancy grid by this point, so
there's no per-track color left to recover. Element ID is binned so each
cell renders as a visible block instead of a single, easy-to-miss pixel.
"""

import os

import numpy as np
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _bin_hits(x2d: np.ndarray, elem_bin: int) -> np.ndarray:
    """Bins the element-ID axis into groups of `elem_bin` and sums hit
    counts, turning the sparse single-pixel-wide binary grid into a coarser
    density grid that renders as clearly visible blocks.

    Args:
        x2d: (num_detectors, num_elements) binary hit matrix.
        elem_bin: number of element IDs summed into one output cell.

    Returns:
        (num_detectors, num_elements // elem_bin) integer hit-count grid.
    """
    num_det, num_elem = x2d.shape
    pad = (-num_elem) % elem_bin
    if pad:
        x2d = np.pad(x2d, ((0, 0), (0, pad)))
    n_bins = x2d.shape[1] // elem_bin
    return x2d.reshape(num_det, n_bins, elem_bin).sum(axis=2)


class EventGalleryCallback(tf.keras.callbacks.Callback):
    """Tracks one fixed validation event across training, saving a heatmap
    PNG of its (binned) hit matrix every `freq` epochs, titled with the
    model's current true-vs-predicted pair count for that event."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str,
        freq: int = 2,
        event_index: int = None,
        elem_bin: int = 6,
        seed: int = 0,
    ):
        super().__init__()
        self._X_val = X_val
        self._y_val = y_val
        self._output_dir = output_dir
        self._freq = freq
        self._elem_bin = elem_bin
        os.makedirs(output_dir, exist_ok=True)

        if event_index is None:
            event_index = int(np.random.default_rng(seed).integers(len(y_val)))
        self._event_index = event_index

    def on_epoch_end(self, epoch, logs=None):
        if not MATPLOTLIB_AVAILABLE or (epoch + 1) % self._freq != 0:
            return

        x = self._X_val[self._event_index]
        t = int(self._y_val[self._event_index])
        pred = self.model.predict(x[None, ...], batch_size=1, verbose=0)
        p = int(np.argmax(pred[0]))

        grid = _bin_hits(x[..., 0], self._elem_bin)  # (num_detectors, n_bins)
        num_det, n_bins = grid.shape

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid.T, aspect="auto", cmap="Reds", origin="lower", interpolation="nearest")
        status = "correct" if t == p else "WRONG"
        ax.set_title(f"Event {self._event_index} -- Epoch {epoch + 1}  (true={t}, pred={p}, {status})")
        ax.set_xlabel(f"Detector ID (1 to {num_det})")
        ax.set_ylabel("Element ID")
        y_ticks = np.linspace(0, n_bins, 6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(v * self._elem_bin)) for v in y_ticks])
        fig.colorbar(im, ax=ax, label="hits per bin")
        fig.tight_layout()
        fig.savefig(os.path.join(self._output_dir, f"epoch_{epoch + 1:03d}.png"))
        plt.close(fig)
