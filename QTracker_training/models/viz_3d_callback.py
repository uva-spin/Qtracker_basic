# ruff: noqa: E402
"""
3D prediction visualizations for MultiTrackFinder.

Generates three plots every `freq` epochs:
  1. Event scatter  — 3D scatter of one event's hit matrix with truth and predicted tracks
  2. Residual waterfall — residual distribution per detector layer (3D histogram)
  3. Layer accuracy  — per-layer exact accuracy by pair and charge (3D bar chart)

Can be used as a Keras callback (add to model.fit callbacks list) or run standalone.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


_PAIR_COLORS = ["#2196F3", "#4CAF50", "#9C27B0"]   # blue, green, purple — truth
_PRED_COLORS = ["#FF5722", "#FF9800", "#E91E63"]    # orange, amber, pink — predicted


class Viz3DCallback(tf.keras.callbacks.Callback):
    """Saves 3D prediction plots every `freq` epochs to `output_dir`."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str = "plots/3d",
        freq: int = 10,
        n_events: int = 3,
        max_pairs: int = 3,
    ):
        """
        Args:
            X_val: noisy hit matrices, shape (N, 62, 201, 1)
            y_val: ground truth element IDs, shape (N, max_pairs, 2, 62)
            output_dir: directory to save PNGs
            freq: generate plots every this many epochs
            n_events: number of individual event scatter plots to save
            max_pairs: number of pair slots
        """
        super().__init__()
        self._X = X_val[:n_events]
        self._y = y_val[:n_events]
        self._X_all = X_val
        self._y_all = y_val
        self._dir = output_dir
        self._freq = freq
        self._n_events = n_events
        self._max_pairs = max_pairs
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if (epoch + 1) % self._freq != 0:
            return

        _, seg_out = self.model.predict(
            tf.cast(self._X_all, tf.float32), verbose=0
        )  # (N, max_pairs, 2, 62, 201)
        y_pred_all = np.argmax(seg_out, axis=-1).astype(np.int32)  # (N, max_pairs, 2, 62)

        for ev in range(self._n_events):
            _plot_event_scatter(
                self._X[ev, :, :, 0],
                self._y[ev],
                y_pred_all[ev],
                save_path=os.path.join(self._dir, f"ep{epoch+1:03d}_ev{ev}_scatter.png"),
            )

        _plot_residual_waterfall(
            self._y_all,
            y_pred_all,
            save_path=os.path.join(self._dir, f"ep{epoch+1:03d}_residuals.png"),
        )

        _plot_layer_accuracy(
            self._y_all,
            y_pred_all,
            self._max_pairs,
            save_path=os.path.join(self._dir, f"ep{epoch+1:03d}_layer_acc.png"),
        )


# ── Plot 1: single-event 3D scatter ──────────────────────────────────────────

def _plot_event_scatter(
    hit_matrix: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """
    3D scatter of one event.

    Args:
        hit_matrix: (62, 201) binary
        y_true: (max_pairs, 2, 62) ground-truth element IDs
        y_pred: (max_pairs, 2, 62) predicted element IDs
    """
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Background hits — subsample for readability
    rows, cols = np.where(hit_matrix > 0)
    rng = np.random.default_rng(0)
    mask = rng.random(len(rows)) < 0.25
    ax.scatter(
        rows[mask], cols[mask], zs=0, zdir="z",
        c="gray", alpha=0.12, s=3, label="All hits",
    )

    n_pairs = y_true.shape[0]
    for p in range(n_pairs):
        z = (p + 1) * 0.4  # stack pairs along z

        # truth — circles
        for c_idx, (label_suffix, marker) in enumerate([("μ+ truth", "o"), ("μ- truth", "^")]):
            layers = np.where(y_true[p, c_idx] != 0)[0]
            elems = y_true[p, c_idx][layers]
            ax.scatter(
                layers, elems, zs=z, zdir="z",
                c=_PAIR_COLORS[p % len(_PAIR_COLORS)], s=18, marker=marker,
                label=f"Pair {p} {label_suffix}" if c_idx == 0 else "_nolegend_",
                alpha=0.85,
            )

        # predictions — crosses, only where truth exists
        for c_idx, (label_suffix, marker) in enumerate([("μ+ pred", "x"), ("μ- pred", "+")]):
            layers = np.where(y_true[p, c_idx] != 0)[0]
            elems = y_pred[p, c_idx][layers]
            ax.scatter(
                layers, elems, zs=z, zdir="z",
                c=_PRED_COLORS[p % len(_PRED_COLORS)], s=22, marker=marker,
                label=f"Pair {p} {label_suffix}" if c_idx == 0 else "_nolegend_",
                alpha=0.9,
            )

    ax.set_xlabel("Detector Layer")
    ax.set_ylabel("Element ID (channel)")
    ax.set_zlabel("Pair (stacked)")
    ax.set_title("Event: Hit Matrix with Truth and Predicted Tracks")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: residual waterfall ────────────────────────────────────────────────

def _plot_residual_waterfall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    n_bins: int = 40,
    layer_stride: int = 3,
) -> None:
    """
    Waterfall histogram: residual distribution for every `layer_stride` detector layers.

    Args:
        y_true: (N, max_pairs, 2, 62)
        y_pred: (N, max_pairs, 2, 62)
    """
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")

    n_layers = y_true.shape[-1]
    residuals = (y_pred - y_true).reshape(-1, n_layers)   # (N*pairs*2, 62)
    valid = y_true.reshape(-1, n_layers) != 0

    bin_edges = np.linspace(-60, 60, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]
    cmap = plt.cm.viridis

    for layer in range(0, n_layers, layer_stride):
        res = residuals[:, layer][valid[:, layer]]
        if len(res) == 0:
            continue
        counts, _ = np.histogram(res, bins=bin_edges)
        counts = counts / (counts.max() + 1e-8)
        ax.bar(
            bin_centers, counts, zs=layer, zdir="y",
            width=width, alpha=0.65,
            color=cmap(layer / n_layers),
        )

    ax.set_xlabel("Residual (channels)")
    ax.set_ylabel("Detector Layer")
    ax.set_zlabel("Normalized Count")
    ax.set_title("Residual Distribution per Detector Layer")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers))
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Layer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Plot 3: per-layer accuracy 3D bar chart ───────────────────────────────────

def _plot_layer_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_pairs: int,
    save_path: str,
) -> None:
    """
    3D bar chart: exact accuracy per detector layer, per pair, per charge.

    Args:
        y_true: (N, max_pairs, 2, 62)
        y_pred: (N, max_pairs, 2, 62)
        max_pairs: number of pair slots
    """
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="3d")

    n_layers = y_true.shape[-1]
    n_series = max_pairs * 2
    layers = np.arange(n_layers)
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_series))

    for idx in range(n_series):
        p = idx // 2
        c = idx % 2
        gt = y_true[:, p, c, :]   # (N, 62)
        pr = y_pred[:, p, c, :]
        valid = gt != 0

        acc = np.zeros(n_layers)
        for l in range(n_layers):
            v = valid[:, l]
            if v.sum() == 0:
                continue
            acc[l] = np.mean((pr[:, l] == gt[:, l])[v])

        label = f"P{p} {'μ+' if c == 0 else 'μ-'}"
        ax.bar(
            layers, acc, zs=idx, zdir="y",
            width=0.85, alpha=0.75,
            color=colors[idx],
            label=label,
        )

    ax.set_xlabel("Detector Layer")
    ax.set_ylabel("Series (pair × charge)")
    ax.set_zlabel("Exact Accuracy")
    ax.set_yticks(range(n_series))
    ax.set_yticklabels(
        [f"P{p} {'μ+' if c==0 else 'μ-'}" for p in range(max_pairs) for c in range(2)],
        fontsize=7,
    )
    ax.set_title("Per-Layer Accuracy by Pair and Charge")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
