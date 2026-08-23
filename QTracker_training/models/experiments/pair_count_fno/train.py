# ruff: noqa: E402
"""
Trains the Axial-FNO pair-count classifier ("Stage A" of the two-stage
classifier + router design).

Consumes the exact same curriculum ROOT files as MultiTrackFinder.py
(low/med/high background-track complexity + validation) -- no new data
generation needed, just a different derived label (see data.py).
"""

import argparse
import gc
import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from model import build_model
from data import load_data_pair_count
from confusion_matrix_callback import ConfusionMatrixCallback
from event_gallery_callback import EventGalleryCallback

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs(os.path.join(_HERE, "checkpoints"), exist_ok=True)
mixed_precision.set_global_policy("mixed_float16")


class MLflowEpochCallback(tf.keras.callbacks.Callback):
    """Log per-epoch metrics via MlflowClient (thread-safe; avoids active_run() thread-local issues)."""

    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id
        self._client = MlflowClient() if MLFLOW_AVAILABLE else None

    def on_epoch_end(self, epoch, logs=None):
        if self._client and logs:
            for key, val in logs.items():
                try:
                    self._client.log_metric(self._run_id, key, float(val), step=epoch)
                except Exception as e:
                    print(f"MLflow metric log failed ({key}={val}): {e}", flush=True)


def _sample_replay(
    X: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator
):
    """Randomly subsample `fraction` of (X, y), to be mixed into a later
    curriculum phase as rehearsal data (see call site in train_model).

    Rehearsal fixes catastrophic forgetting: the high phase trains on
    34-50-background-track events only, for many epochs, with no exposure to
    easier events -- so the model drifts away from what it learned about
    low/med-complexity events, even though val is representative across the
    full range and doesn't change. Mixing a slice of the earlier phase's data
    into the high phase's training set keeps those gradient updates present.
    """
    if fraction <= 0:
        return None, None
    n = int(len(y) * fraction)
    if n <= 0:
        return None, None
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _class_weights(y: np.ndarray, num_classes: int) -> dict:
    """Inverse-frequency class weights, normalized so the average weight is 1.

    Pair-slot counts are known to be imbalanced (pair 2 has far fewer valid
    events than pair 0 in the joint MultiTrackFinder data), so the same
    imbalance is expected here across the n=0..max_pairs label.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for absent classes
    inv = 1.0 / counts
    inv *= num_classes / inv.sum()
    return {i: float(w) for i, w in enumerate(inv)}


def _fit_phase(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    initial_epoch: int,
    epochs: int,
    batch_size: int,
    lr: float,
    callbacks_extra: list,
    num_classes: int,
    label: str,
):
    print(f"\n=== Phase: {label} ({len(y)} events, epochs {initial_epoch}->{epochs}) ===", flush=True)
    counts = np.bincount(y, minlength=num_classes)
    print(f"Class distribution: {dict(enumerate(counts.tolist()))}", flush=True)
    class_weight = _class_weights(y, num_classes)
    print(f"Class weights: {class_weight}", flush=True)

    cur_lr = float(model.optimizer.learning_rate.numpy())
    target_lr = min(lr, cur_lr * 3.0) if initial_epoch > 0 else lr
    model.optimizer.learning_rate.assign(target_lr)
    print(f"LR -> {target_lr:.2e} (target {lr:.2e})", flush=True)

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    return model.fit(
        X,
        y,
        initial_epoch=initial_epoch,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[lr_scheduler, *callbacks_extra],
        verbose=2,
    )


def train_model(args: argparse.Namespace) -> None:
    # Ensure the actual output directory exists -- the module-level
    # os.makedirs(_HERE/checkpoints) only covers the CODEDIR-relative
    # default; --output_model may point elsewhere (e.g. persistent project
    # storage, as train.slurm now does).
    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(vars(args))

    active_run_id = (
        mlflow.active_run().info.run_id if (MLFLOW_AVAILABLE and mlflow.active_run()) else None
    )
    mlflow_cb = MLflowEpochCallback(run_id=active_run_id) if active_run_id else tf.keras.callbacks.Callback()

    num_classes = args.max_pairs + 1

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)} -- {[g.name for g in gpus]}", flush=True)

    model = build_model(
        max_pairs=args.max_pairs,
        base=args.base,
        fno_depth=args.fno_depth,
        k_max=args.k_max,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model.summary()

    optimizer = AdamW(learning_rate=args.lr_low, weight_decay=args.weight_decay, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[SparseCategoricalAccuracy(name="acc")],
    )

    X_val, y_val = load_data_pair_count(args.val_root_file, args.max_pairs)
    if X_val is None:
        return

    best_ckpt_path = args.output_model.replace(".keras", "_best.keras")
    checkpoint = ModelCheckpoint(best_ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

    confusion_dir = os.path.join(os.path.dirname(args.output_model), "plots", "confusion")
    confusion_cb = ConfusionMatrixCallback(
        X_val=X_val, y_val=y_val, output_dir=confusion_dir,
        num_classes=num_classes, freq=args.confusion_freq,
    )

    gallery_dir = os.path.join(os.path.dirname(args.output_model), "plots", "gallery")
    gallery_cb = EventGalleryCallback(
        X_val=X_val, y_val=y_val, output_dir=gallery_dir, freq=args.confusion_freq,
    )

    # Load every complexity level present, keyed by its real identity --
    # curriculum *order* is decided separately below, via --reverse_curriculum,
    # so a phase's label always matches the data it actually trains on.
    named_phases = [("low", args.train_root_file_low, args.lr_low)]
    if args.train_root_file_med:
        named_phases.append(("med", args.train_root_file_med, args.lr_med))
    if args.train_root_file_high:
        named_phases.append(("high", args.train_root_file_high, args.lr_high))

    loaded = []
    for label, path, lr in named_phases:
        X, y = load_data_pair_count(path, args.max_pairs)
        if X is None:
            return
        loaded.append((label, X, y, lr))

    # Default order is low -> med -> high (easy first). --reverse_curriculum
    # flips it to high -> med -> low, to test whether starting on noisier
    # data suppresses the overfitting a high-capacity model shows when it
    # sees clean, low-complexity data first with nothing to regularize it.
    if args.reverse_curriculum:
        loaded = list(reversed(loaded))

    # Epoch boundaries: same low_ratio/med_ratio-of-total-epochs split as
    # before, just applied to however many phases are actually present.
    n_phases = len(loaded)
    if n_phases == 3:
        bounds = [0, int(args.epochs * args.low_ratio), int(args.epochs * args.med_ratio), args.epochs]
    elif n_phases == 2:
        bounds = [0, int(args.epochs * args.low_ratio), args.epochs]
    else:
        bounds = [0, args.epochs]

    replay_rng = np.random.default_rng(42)
    replay_X: list = []
    replay_y: list = []

    for i, (label, X, y, lr) in enumerate(loaded):
        is_last = i == n_phases - 1
        callbacks = [checkpoint, mlflow_cb, confusion_cb, gallery_cb]

        if is_last:
            callbacks = [checkpoint, early_stopping, mlflow_cb, confusion_cb, gallery_cb]
            if replay_X:
                n_replay = sum(len(a) for a in replay_y)
                print(
                    f"{label} phase rehearsal: mixing in {n_replay} earlier-phase events "
                    f"(replay_fraction={args.replay_fraction})", flush=True,
                )
                X = np.concatenate([X, *replay_X], axis=0)
                y = np.concatenate([y, *replay_y], axis=0)
        else:
            rX, ry = _sample_replay(X, y, args.replay_fraction, replay_rng)
            if rX is not None:
                replay_X.append(rX)
                replay_y.append(ry)

        _fit_phase(
            model, X, y, X_val, y_val, bounds[i], bounds[i + 1], args.batch_size,
            lr, callbacks, num_classes, label,
        )
        del X, y
        gc.collect()

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_artifact(args.output_model, artifact_path="model")
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Axial-FNO pair-count classifier.")
    parser.add_argument("train_root_file_low", type=str, help="Path to the low-complexity train ROOT file.")
    parser.add_argument("val_root_file", type=str, help="Path to the validation ROOT file.")
    parser.add_argument("--train_root_file_med", type=str, default=None)
    parser.add_argument("--train_root_file_high", type=str, default=None)
    parser.add_argument(
        "--output_model",
        type=str,
        default=os.path.join(_HERE, "checkpoints", "pair_count_fno.keras"),
    )
    parser.add_argument("--max_pairs", type=int, default=3, help="Max dimuon pairs; classes are 0..max_pairs.")
    parser.add_argument("--base", type=int, default=32, help="Hidden channel width.")
    parser.add_argument("--fno_depth", type=int, default=4, help="Number of Fourier+attention block pairs.")
    parser.add_argument("--k_max", type=int, default=32, help="Fourier modes kept per block.")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads for detector-axis mixing.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr_low", type=float, default=3e-4)
    parser.add_argument("--lr_med", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--low_ratio", type=float, default=0.5)
    parser.add_argument("--med_ratio", type=float, default=0.8)
    parser.add_argument(
        "--replay_fraction", type=float, default=0.15,
        help="Fraction of each of low/med kept as rehearsal data mixed into the high phase (0 disables).",
    )
    parser.add_argument(
        "--confusion_freq", type=int, default=2,
        help="Save a validation confusion-matrix PNG every N epochs (frames for make_gif.py).",
    )
    parser.add_argument(
        "--reverse_curriculum", action="store_true",
        help="Train high -> med -> low instead of low -> med -> high. Rehearsal "
             "replay direction flips automatically (always earlier phases -> last phase).",
    )
    parser.add_argument("--mlflow_experiment", type=str, default="pair_count_fno")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    args = parser.parse_args()

    train_model(args)
