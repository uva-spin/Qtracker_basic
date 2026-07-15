# ruff: noqa: E402

import argparse
import gc
import itertools
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall

from backbones import unetpp_backbone
from data_loader import load_data_denoise
from losses import weighted_bce, EPSILON, OVERLAP_LAMBDA

try:
    import mlflow
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs("checkpoints", exist_ok=True)
mixed_precision.set_global_policy("mixed_float16")

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


# ---------------------------------------------------------------------------
# Callbacks / metrics (reused from MultiTrackFinder)
# ---------------------------------------------------------------------------

class MLflowEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id
        self._client = MlflowClient() if MLFLOW_AVAILABLE else None

    def on_epoch_end(self, epoch, logs=None):
        if self._client and logs:
            for key, val in logs.items():
                try:
                    self._client.log_metric(self._run_id, key, float(val), step=epoch)
                except Exception:
                    pass


class LivePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path: str, every_n: int = 5):
        super().__init__()
        self._path = output_path
        self._every_n = every_n
        self._history: dict[str, list] = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for k, v in logs.items():
                self._history.setdefault(k, []).append(float(v))
        if not MATPLOTLIB_AVAILABLE or (epoch + 1) % self._every_n != 0:
            return
        epochs = range(1, len(self._history.get("loss", [])) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (train_key, val_key, title) in zip(axes, [
            ("loss", "val_loss", "Total Loss"),
            ("segment_segment_nonempty_acc", "val_segment_segment_nonempty_acc", "Non-Empty Accuracy"),
            ("segment_segment_mean_residual", "val_segment_segment_mean_residual", "Mean Residual (ch)"),
        ]):
            if train_key in self._history:
                ax.plot(epochs, self._history[train_key], label="train")
            if val_key in self._history:
                ax.plot(epochs, self._history[val_key], label="val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(self._path)
        plt.close(fig)


class EnsembleCheckpoint(tf.keras.callbacks.Callback):
    """Save each sub-model independently when val_loss improves."""

    def __init__(self, sub_models: list, output_dir: str, monitor: str = "val_loss"):
        super().__init__()
        self.sub_models = sub_models
        self.output_dir = output_dir
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor, float("inf"))
        if current < self.best:
            self.best = current
            for i, m in enumerate(self.sub_models):
                path = os.path.join(self.output_dir, f"ensemble_model_{i}_best.keras")
                m.save(path)
            print(f"\nEnsemble best saved (val_loss={current:.4f})", flush=True)


class SegmentNonEmptyAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="segment_nonempty_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self._correct = self.add_weight(name="correct", initializer="zeros")
        self._total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        non_empty = tf.not_equal(y_true, 0)
        correct = tf.logical_and(tf.equal(pred_ids, y_true), non_empty)
        self._correct.assign_add(tf.cast(tf.reduce_sum(tf.cast(correct, tf.int32)), tf.float32))
        self._total.assign_add(tf.cast(tf.reduce_sum(tf.cast(non_empty, tf.int32)), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._correct, self._total)

    def reset_state(self):
        self._correct.assign(0.0)
        self._total.assign(0.0)


class SegmentMeanResidual(tf.keras.metrics.Metric):
    def __init__(self, name="segment_mean_residual", **kwargs):
        super().__init__(name=name, **kwargs)
        self._residual_sum = self.add_weight(name="residual_sum", initializer="zeros")
        self._total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        non_empty = tf.not_equal(y_true, 0.0)
        residual = tf.abs(pred_ids - y_true)
        self._residual_sum.assign_add(tf.reduce_sum(tf.where(non_empty, residual, 0.0)))
        self._total.assign_add(tf.cast(tf.reduce_sum(tf.cast(non_empty, tf.int32)), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._residual_sum, self._total)

    def reset_state(self):
        self._residual_sum.assign(0.0)
        self._total.assign(0.0)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_single_model(
    num_detectors: int = 62,
    num_elementIDs: int = 201,
    use_bn: bool = False,
    dropout_bn: float = 0.0,
    dropout_enc: float = 0.0,
    denoise_base: int = 24,
    base: int = 32,
    use_attn: bool = False,
    use_attn_ffn: bool = True,
    dropout_attn: float = 0.0,
    model_idx: int = 0,
) -> tf.keras.Model:
    """Single-pair U-Net++ denoiser + segmenter (n_pairs=1, independent weights per model)."""
    p = f"m{model_idx}_"
    inp = layers.Input(shape=(num_detectors, num_elementIDs, 1))
    x = unetpp_backbone(inp, num_detectors, num_elementIDs, use_bn, dropout_bn, dropout_enc, denoise_base, use_attn=False)
    denoise_out = layers.Conv2D(1, kernel_size=1, name=f"{p}denoise", dtype=tf.float32)(x)
    x = unetpp_backbone(denoise_out, num_detectors, num_elementIDs, use_bn, dropout_bn, dropout_enc, base, use_attn, use_attn_ffn, dropout_attn)
    x = layers.Conv2D(2, kernel_size=1, name=f"{p}seg_conv")(x)           # (B, 62, 201, 2)
    x = layers.Permute((3, 1, 2), name=f"{p}seg_perm")(x)                 # (B, 2, 62, 201)
    x = layers.Reshape((1, 2, num_detectors, num_elementIDs), name=f"{p}seg_reshape")(x)  # (B,1,2,62,201)
    seg_out = layers.Softmax(axis=-1, name=f"{p}segment", dtype=tf.float32)(x)
    return tf.keras.Model(inputs=inp, outputs=[denoise_out, seg_out], name=f"ensemble_model_{model_idx}")


class EnsembleWrapper(tf.keras.Model):
    """
    Wraps n_models independent single-pair models into one jointly-trainable unit.

    Outputs:
        "denoise": averaged denoised output (B, 62, 201, 1)
        "segment": stacked per-model predictions (B, n_models, 2, 62, 201)
    """

    def __init__(self, sub_models: list, **kwargs):
        super().__init__(**kwargs)
        self.sub_models = sub_models

    def call(self, x, training=False):
        denoise_outs, seg_outs = [], []
        for m in self.sub_models:
            d, s = m(x, training=training)
            denoise_outs.append(d)       # (B, 62, 201, 1)
            seg_outs.append(s)           # (B, 1, 2, 62, 201)
        denoise_avg = tf.add_n(denoise_outs) / len(self.sub_models)  # (B, 62, 201, 1)
        seg_stacked = tf.concat(seg_outs, axis=1)                    # (B, n_models, 2, 62, 201)
        return {"denoise": denoise_avg, "segment": seg_stacked}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def ensemble_seg_loss(n_models: int, diversity_lambda: float = 0.05):
    """
    Permutation-invariant segmentation loss across N independent models + diversity penalty.

    y_true: (B, n_models, 2, 62) — GT pair labels (n_models == max_pairs expected)
    y_pred: (B, n_models, 2, 62, 201) — stacked per-model softmax outputs

    Permutation term: identical to min_perm_loss but applied across model outputs.
    Diversity term: minimises inner product between each pair of model outputs,
                    pushing models towards different tracks.
    """
    perms = list(itertools.permutations(range(n_models)))
    perm_tensors = [tf.constant(list(p), dtype=tf.int32) for p in perms]

    def loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.int32)
        n = n_models

        pred_tiled = tf.tile(tf.expand_dims(y_pred, axis=2), [1, 1, n, 1, 1, 1])  # (B,N,N,2,62,201)
        true_tiled = tf.tile(tf.expand_dims(y_true, axis=1), [1, n, 1, 1, 1])     # (B,N,N,2,62)
        ce = tf.keras.losses.sparse_categorical_crossentropy(true_tiled, pred_tiled)  # (B,N,N,2,62)
        cost = tf.reduce_mean(ce, axis=[3, 4])  # (B, N, N)

        perm_costs = []
        for perm_tensor in perm_tensors:
            cost_perm = tf.gather(cost, perm_tensor, axis=2)
            perm_costs.append(tf.linalg.trace(cost_perm))
        seg_loss = tf.reduce_mean(tf.reduce_min(tf.stack(perm_costs, axis=1), axis=1))

        # Diversity: minimise inner product of softmax dists between every model pair
        div_loss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                si = y_pred[:, i]  # (B, 2, 62, 201)
                sj = y_pred[:, j]
                overlap = tf.reduce_mean(tf.reduce_sum(si * sj, axis=-1))
                div_loss += overlap

        return seg_loss + diversity_lambda * div_loss

    return loss


def _plot_loss_curves(all_history: list[dict], output_path: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    train_loss, val_loss = [], []
    for h in all_history:
        train_loss.extend(h.get("loss", []))
        val_loss.extend(h.get("val_loss", []))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(train_loss) + 1), train_loss, label="train loss")
    ax.plot(range(1, len(val_loss) + 1), val_loss, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Ensemble Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(getattr(args, "mlflow_experiment", "ensemble_track_training"))
        mlflow.start_run(run_name=getattr(args, "mlflow_run_name", None))
        mlflow.log_params(vars(args))

    _active_run_id = mlflow.active_run().info.run_id if (MLFLOW_AVAILABLE and mlflow.active_run()) else None
    mlflow_cb = MLflowEpochCallback(run_id=_active_run_id) if _active_run_id else tf.keras.callbacks.Callback()
    _all_histories: list[dict] = []

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build n_models independent single-pair models
    sub_models = [
        build_single_model(
            num_detectors=NUM_DETECTORS,
            num_elementIDs=NUM_ELEMENT_IDS,
            use_bn=args.batch_norm,
            dropout_bn=args.dropout_bn,
            dropout_enc=args.dropout_enc,
            denoise_base=args.denoise_base,
            base=args.base,
            use_attn=args.use_attn,
            use_attn_ffn=args.use_attn_ffn,
            dropout_attn=args.dropout_attn,
            model_idx=i,
        )
        for i in range(args.n_models)
    ]
    print(f"Built {args.n_models} independent single-pair models", flush=True)
    sub_models[0].summary()

    ensemble = EnsembleWrapper(sub_models, name="ensemble")

    optimizer = AdamW(
        learning_rate=args.lr_low,
        weight_decay=args.weight_decay,
        clipnorm=args.clipnorm,
    )

    ensemble.compile(
        optimizer=optimizer,
        loss={
            "denoise": weighted_bce(pos_weight=args.pos_weight),
            "segment": ensemble_seg_loss(n_models=args.n_models, diversity_lambda=args.diversity_lambda),
        },
        loss_weights={"denoise": 3.0, "segment": 1.0},
        metrics={
            "denoise": [Precision(name="precision"), Recall(name="recall")],
            "segment": [SegmentNonEmptyAccuracy(), SegmentMeanResidual()],
        },
    )

    # Data loading
    X_train_low, X_clean_train_low, y_mup_low, y_mum_low = load_data_denoise(
        args.train_root_file_low, multi_track=True, max_pairs=args.n_models
    )
    if X_train_low is None:
        return
    y_train_low = np.stack([y_mup_low, y_mum_low], axis=2)  # (N, n_models, 2, 62)

    X_val, X_clean_val, y_mup_val, y_mum_val = load_data_denoise(
        args.val_root_file, multi_track=True, max_pairs=args.n_models
    )
    if X_val is None:
        return
    y_val = np.stack([y_mup_val, y_mum_val], axis=2)

    def _fit(X, X_clean, y_seg, initial_epoch, target_epoch, callbacks):
        return ensemble.fit(
            X,
            {"denoise": X_clean, "segment": y_seg},
            initial_epoch=initial_epoch,
            epochs=target_epoch,
            batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=callbacks,
            verbose=2,
        )

    live_plot_path = os.path.join(args.output_dir, "training_progress.png")
    live_plot_cb = LivePlotCallback(output_path=live_plot_path, every_n=5)
    ens_ckpt = EnsembleCheckpoint(sub_models, args.output_dir, monitor="val_loss")

    if args.train_root_file_med and args.train_root_file_high:
        print("Curriculum learning enabled.", flush=True)
        epochs_low = int(args.epochs * args.low_ratio)
        epochs_med = int(args.epochs * args.med_ratio)

        lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        hist = _fit(X_train_low, X_clean_train_low, y_train_low, 0, epochs_low,
                    [lr_sched, ens_ckpt, mlflow_cb, live_plot_cb])
        _all_histories.append(hist.history)
        del X_train_low, X_clean_train_low, y_train_low; gc.collect()

        X_train_med, X_clean_med, y_mup_med, y_mum_med = load_data_denoise(
            args.train_root_file_med, multi_track=True, max_pairs=args.n_models
        )
        y_train_med = np.stack([y_mup_med, y_mum_med], axis=2)
        _cur_lr = float(ensemble.optimizer.learning_rate.numpy())
        _lr_med = min(args.lr_med, _cur_lr * 3.0)
        print(f"LR transition low→med: {_cur_lr:.2e} → {_lr_med:.2e} (target {args.lr_med:.2e})", flush=True)
        ensemble.optimizer.learning_rate.assign(_lr_med)
        lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        hist = _fit(X_train_med, X_clean_med, y_train_med, epochs_low, epochs_med,
                    [lr_sched, ens_ckpt, mlflow_cb, live_plot_cb])
        _all_histories.append(hist.history)
        del X_train_med, X_clean_med, y_train_med; gc.collect()

        X_train_high, X_clean_high, y_mup_high, y_mum_high = load_data_denoise(
            args.train_root_file_high, multi_track=True, max_pairs=args.n_models
        )
        y_train_high = np.stack([y_mup_high, y_mum_high], axis=2)
        _cur_lr = float(ensemble.optimizer.learning_rate.numpy())
        _lr_high = min(args.lr_high, _cur_lr * 3.0)
        print(f"LR transition med→high: {_cur_lr:.2e} → {_lr_high:.2e} (target {args.lr_high:.2e})", flush=True)
        ensemble.optimizer.learning_rate.assign(_lr_high)
        lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=False)
        hist = _fit(X_train_high, X_clean_high, y_train_high, epochs_med, args.epochs,
                    [lr_sched, early_stop, ens_ckpt, mlflow_cb, live_plot_cb])
        _all_histories.append(hist.history)
        del X_train_high, X_clean_high, y_train_high; gc.collect()

    else:
        lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=False)
        hist = _fit(X_train_low, X_clean_train_low, y_train_low, 0, args.epochs,
                    [lr_sched, early_stop, ens_ckpt, mlflow_cb, live_plot_cb])
        _all_histories.append(hist.history)

    # Save final sub-models
    for i, m in enumerate(sub_models):
        path = os.path.join(args.output_dir, f"ensemble_model_{i}.keras")
        m.save(path)
        print(f"Saved sub-model {i} to {path}", flush=True)

    if MLFLOW_AVAILABLE and mlflow.active_run():
        loss_plot = os.path.join(args.output_dir, "loss_curves.png")
        _plot_loss_curves(_all_histories, loss_plot)
        if os.path.exists(loss_plot):
            mlflow.log_artifact(loss_plot, artifact_path="plots")
        if os.path.exists(live_plot_path):
            mlflow.log_artifact(live_plot_path, artifact_path="plots")
        mlflow.end_run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble of independent single-pair track finders.")
    parser.add_argument("train_root_file_low", type=str)
    parser.add_argument("val_root_file", type=str)
    parser.add_argument("--train_root_file_med", type=str, default=None)
    parser.add_argument("--train_root_file_high", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/ensemble",
                        help="Directory to save individual sub-model checkpoints.")
    parser.add_argument("--n_models", type=int, default=3,
                        help="Number of independent single-pair models in the ensemble.")
    parser.add_argument("--diversity_lambda", type=float, default=0.05,
                        help="Weight for cross-model diversity penalty (encourages models to find different tracks).")
    parser.add_argument("--lr_low", type=float, default=3e-4)
    parser.add_argument("--lr_med", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=3e-5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--use_attn", type=int, default=1)
    parser.add_argument("--use_attn_ffn", type=int, default=0)
    parser.add_argument("--dropout_bn", type=float, default=0.5)
    parser.add_argument("--dropout_enc", type=float, default=0.2)
    parser.add_argument("--dropout_attn", type=float, default=0.1)
    parser.add_argument("--denoise_base", type=int, default=24)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=20.0)
    parser.add_argument("--low_ratio", type=float, default=0.5)
    parser.add_argument("--med_ratio", type=float, default=0.7)
    parser.add_argument("--mlflow_experiment", type=str, default="ensemble_track_v1")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    args = parser.parse_args()
    train_model(args)
