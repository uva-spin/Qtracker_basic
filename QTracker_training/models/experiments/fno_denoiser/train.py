# ruff: noqa: E402
"""
Trains the FNO denoiser standalone (denoise loss only, no segmentation head)
against the same multi-track curriculum ROOT files MultiTrackFinder.py uses,
so denoise_precision/denoise_recall are directly comparable to that model's
numbers at a similar point in training.

Curriculum rehearsal (mixing ~15% of low/med data into the high phase) is
included from the start this time -- already validated on the pair-count
classifier and ported to MultiTrackFinder.py; no reason to rediscover the
same curriculum-forgetting failure a third time.
"""

import argparse
import gc
import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import numpy as np
import ROOT  # noqa: F401
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall

from model import build_denoiser
from data_loader import load_data_denoise  # noqa: E402
from losses import weighted_bce  # noqa: E402

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

tf.random.set_seed(42)
np.random.seed(42)
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


def _sample_replay(X: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator):
    """Randomly subsample `fraction` of (X, y) as curriculum rehearsal data
    mixed into the high phase -- see MultiTrackFinder.py's identical helper
    for why this is needed."""
    if fraction <= 0:
        return None, None
    n = int(len(y) * fraction)
    if n <= 0:
        return None, None
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _load_denoise_pair(root_file: str, max_pairs: int):
    """load_data_denoise returns (X, X_clean, y_muPlus, y_muMinus); this
    experiment only needs X/X_clean, the muon-pair labels are unused."""
    X, X_clean, _y_muPlus, _y_muMinus = load_data_denoise(
        root_file, multi_track=True, max_pairs=max_pairs
    )
    return X, X_clean


def train_model(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(vars(args))

    active_run_id = (
        mlflow.active_run().info.run_id if (MLFLOW_AVAILABLE and mlflow.active_run()) else None
    )
    mlflow_cb = MLflowEpochCallback(run_id=active_run_id) if active_run_id else tf.keras.callbacks.Callback()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)} -- {[g.name for g in gpus]}", flush=True)

    model = build_denoiser(
        base=args.base, fno_depth=args.fno_depth, k_max=args.k_max, num_heads=args.num_heads,
    )
    model.summary()

    optimizer = AdamW(learning_rate=args.lr_low, weight_decay=args.weight_decay, clipnorm=args.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss=weighted_bce(pos_weight=args.pos_weight),
        metrics=[Precision(name="precision"), Recall(name="recall")],
    )

    X_val, Xc_val = _load_denoise_pair(args.val_root_file, args.max_pairs)
    if X_val is None:
        return

    best_ckpt_path = args.output_model.replace(".keras", "_best.keras")
    checkpoint = ModelCheckpoint(best_ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)

    named_phases = [("low", args.train_root_file_low, args.lr_low)]
    if args.train_root_file_med:
        named_phases.append(("med", args.train_root_file_med, args.lr_med))
    if args.train_root_file_high:
        named_phases.append(("high", args.train_root_file_high, args.lr_high))

    loaded = []
    for label, path, lr in named_phases:
        X, Xc = _load_denoise_pair(path, args.max_pairs)
        if X is None:
            return
        loaded.append((label, X, Xc, lr))

    n_phases = len(loaded)
    if n_phases == 3:
        bounds = [0, int(args.epochs * args.low_ratio), int(args.epochs * args.med_ratio), args.epochs]
    elif n_phases == 2:
        bounds = [0, int(args.epochs * args.low_ratio), args.epochs]
    else:
        bounds = [0, args.epochs]

    replay_rng = np.random.default_rng(42)
    replay_X: list = []
    replay_Xc: list = []

    for i, (label, X, Xc, lr) in enumerate(loaded):
        is_last = i == n_phases - 1
        callbacks = [checkpoint, mlflow_cb]

        if is_last:
            callbacks = [checkpoint, early_stopping, mlflow_cb]
            if replay_X:
                n_replay = sum(len(a) for a in replay_X)
                print(f"{label} phase rehearsal: mixing in {n_replay} earlier-phase events "
                      f"(replay_fraction={args.replay_fraction})", flush=True)
                X = np.concatenate([X, *replay_X], axis=0)
                Xc = np.concatenate([Xc, *replay_Xc], axis=0)
        else:
            rX, rXc = _sample_replay(X, Xc, args.replay_fraction, replay_rng)
            if rX is not None:
                replay_X.append(rX)
                replay_Xc.append(rXc)

        print(f"\n=== Phase: {label} ({len(Xc)} events, epochs {bounds[i]}->{bounds[i + 1]}) ===", flush=True)
        cur_lr = float(model.optimizer.learning_rate.numpy())
        target_lr = min(lr, cur_lr * 3.0) if i > 0 else lr
        model.optimizer.learning_rate.assign(target_lr)
        print(f"LR -> {target_lr:.2e} (target {lr:.2e})", flush=True)

        lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        model.fit(
            X, Xc,
            initial_epoch=bounds[i],
            epochs=bounds[i + 1],
            batch_size=args.batch_size,
            validation_data=(X_val, Xc_val),
            callbacks=[lr_scheduler, *callbacks],
            verbose=2,
        )
        del X, Xc
        gc.collect()

    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_artifact(args.output_model, artifact_path="model")
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FNO denoiser.")
    parser.add_argument("train_root_file_low", type=str)
    parser.add_argument("val_root_file", type=str)
    parser.add_argument("--train_root_file_med", type=str, default=None)
    parser.add_argument("--train_root_file_high", type=str, default=None)
    parser.add_argument("--output_model", type=str, default=os.path.join(_HERE, "checkpoints", "fno_denoiser.keras"))
    parser.add_argument("--max_pairs", type=int, default=3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--fno_depth", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr_low", type=float, default=3e-4)
    parser.add_argument("--lr_med", type=float, default=1e-4)
    parser.add_argument("--lr_high", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=20.0, help="Matches MultiTrackFinder.py's denoiser default.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--low_ratio", type=float, default=0.5)
    parser.add_argument("--med_ratio", type=float, default=0.8)
    parser.add_argument("--replay_fraction", type=float, default=0.15)
    parser.add_argument("--mlflow_experiment", type=str, default="fno_denoiser")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    args = parser.parse_args()

    train_model(args)
