# ruff: noqa: E402
"""
Ensemble of independent single-pair models (EnsembleTrackFinder.py's
architecture: n_models fully separate sub-models, not a shared backbone),
but with each sub-model's U-Net++ denoiser replaced by the FNO denoiser
(models/experiments/fno_denoiser/model.py) -- full (62,201) resolution
throughout, no encoder/decoder.

Imports EnsembleWrapper, ensemble_seg_loss, the segment metrics, and the
checkpoint/plotting callbacks directly from EnsembleTrackFinder.py (read-only
-- that file is not modified). Only build_single_model is replaced locally;
the segmenter half of each sub-model is untouched U-Net++ + AxialAttention,
so the denoiser swap is the one isolated variable against the existing
ensemble result (18%/12% accuracy, 34/27-channel residual, halved capacity
+ diversity penalty -- see EXPERIMENTS.md). This run also uses full capacity
(base=32 to match, not halved further) and keeps the same diversity_lambda
default so the denoiser swap is the only new variable versus that baseline.

Curriculum rehearsal (validated on the classifier, ported to
MultiTrackFinder.py) is included from the start.
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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import Precision, Recall

from backbones import unetpp_backbone  # noqa: E402
from data_loader import load_data_denoise  # noqa: E402
from losses import weighted_bce  # noqa: E402
from fno_layers import FourierBlock1D  # noqa: E402
from layers import AxialAttention  # noqa: E402

from EnsembleTrackFinder import (  # noqa: E402 -- read-only reuse, not modified
    EnsembleWrapper,
    EnsembleCheckpoint,
    SegmentNonEmptyAccuracy,
    SegmentMeanResidual,
    LivePlotCallback,
    MLflowEpochCallback,
    ensemble_seg_loss,
    _plot_loss_curves,
    MLFLOW_AVAILABLE,
    NUM_DETECTORS,
    NUM_ELEMENT_IDS,
)

if MLFLOW_AVAILABLE:
    import mlflow

tf.random.set_seed(42)
np.random.seed(42)


def build_single_model_fno_denoiser(
    num_detectors: int = NUM_DETECTORS,
    num_elementIDs: int = NUM_ELEMENT_IDS,
    use_bn: bool = False,
    dropout_bn: float = 0.0,
    dropout_enc: float = 0.0,
    base: int = 32,
    use_attn: bool = False,
    use_attn_ffn: bool = True,
    dropout_attn: float = 0.0,
    fno_base: int = 32,
    fno_depth: int = 4,
    k_max: int = 32,
    num_heads: int = 4,
    model_idx: int = 0,
) -> tf.keras.Model:
    """Single-pair FNO-denoiser + U-Net++ segmenter (n_pairs=1, independent
    weights per model). Same segmenter half as EnsembleTrackFinder.py's
    build_single_model; only the denoiser is replaced."""
    p = f"m{model_idx}_"
    inp = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    x = layers.Conv2D(fno_base, kernel_size=1, name=f"{p}fno_lift")(inp)
    for i in range(fno_depth):
        x = FourierBlock1D(channels=fno_base, k_max=k_max, name=f"{p}fourier_block_{i}")(x)
        x = AxialAttention(
            embed_dim=fno_base, num_heads=num_heads, axis="height", use_ffn=False,
            name=f"{p}detector_mix_{i}",
        )(x)
    denoise_out = layers.Conv2D(1, kernel_size=1, name=f"{p}denoise", dtype=tf.float32)(x)

    x = unetpp_backbone(denoise_out, num_detectors, num_elementIDs, use_bn, dropout_bn, dropout_enc, base, use_attn, use_attn_ffn, dropout_attn)
    x = layers.Conv2D(2, kernel_size=1, name=f"{p}seg_conv")(x)
    x = layers.Permute((3, 1, 2), name=f"{p}seg_perm")(x)
    x = layers.Reshape((1, 2, num_detectors, num_elementIDs), name=f"{p}seg_reshape")(x)
    seg_out = layers.Softmax(axis=-1, name=f"{p}segment", dtype=tf.float32)(x)
    return tf.keras.Model(inputs=inp, outputs=[denoise_out, seg_out], name=f"ensemble_fno_model_{model_idx}")


def _sample_replay(X: np.ndarray, X_clean: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator):
    """Same rehearsal helper as MultiTrackFinder.py / pair_count_fno's train.py."""
    if fraction <= 0:
        return None, None, None
    n = int(len(y) * fraction)
    if n <= 0:
        return None, None, None
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], X_clean[idx], y[idx]


def train_model(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(vars(args))

    _active_run_id = mlflow.active_run().info.run_id if (MLFLOW_AVAILABLE and mlflow.active_run()) else None
    mlflow_cb = MLflowEpochCallback(run_id=_active_run_id) if _active_run_id else tf.keras.callbacks.Callback()
    _all_histories: list = []

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}", flush=True)

    sub_models = [
        build_single_model_fno_denoiser(
            use_bn=args.batch_norm, dropout_bn=args.dropout_bn, dropout_enc=args.dropout_enc,
            base=args.base, use_attn=args.use_attn, use_attn_ffn=args.use_attn_ffn,
            dropout_attn=args.dropout_attn, fno_base=args.fno_base, fno_depth=args.fno_depth,
            k_max=args.k_max, num_heads=args.num_heads, model_idx=i,
        )
        for i in range(args.n_models)
    ]
    print(f"Built {args.n_models} independent single-pair models (FNO denoiser)", flush=True)
    sub_models[0].summary()

    ensemble = EnsembleWrapper(sub_models, name="ensemble_fno")
    optimizer = AdamW(learning_rate=args.lr_low, weight_decay=args.weight_decay, clipnorm=args.clipnorm)
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

    X_val, X_clean_val, y_mup_val, y_mum_val = load_data_denoise(args.val_root_file, multi_track=True, max_pairs=args.n_models)
    if X_val is None:
        return
    y_val = np.stack([y_mup_val, y_mum_val], axis=2)

    live_plot_path = os.path.join(args.output_dir, "training_progress.png")
    live_plot_cb = LivePlotCallback(output_path=live_plot_path, every_n=5)
    ens_ckpt = EnsembleCheckpoint(sub_models, args.output_dir, monitor="val_loss")

    def _fit(X, X_clean, y_seg, initial_epoch, target_epoch, callbacks):
        return ensemble.fit(
            X, {"denoise": X_clean, "segment": y_seg},
            initial_epoch=initial_epoch, epochs=target_epoch, batch_size=args.batch_size,
            validation_data=(X_val, {"denoise": X_clean_val, "segment": y_val}),
            callbacks=callbacks, verbose=2,
        )

    named_phases = [("low", args.train_root_file_low, args.lr_low)]
    if args.train_root_file_med:
        named_phases.append(("med", args.train_root_file_med, args.lr_med))
    if args.train_root_file_high:
        named_phases.append(("high", args.train_root_file_high, args.lr_high))

    loaded = []
    for label, path, lr in named_phases:
        X, X_clean, y_mup, y_mum = load_data_denoise(path, multi_track=True, max_pairs=args.n_models)
        if X is None:
            return
        y_seg = np.stack([y_mup, y_mum], axis=2)
        loaded.append((label, X, X_clean, y_seg, lr))

    n_phases = len(loaded)
    if n_phases == 3:
        bounds = [0, int(args.epochs * args.low_ratio), int(args.epochs * args.med_ratio), args.epochs]
    elif n_phases == 2:
        bounds = [0, int(args.epochs * args.low_ratio), args.epochs]
    else:
        bounds = [0, args.epochs]

    replay_rng = np.random.default_rng(42)
    replay_X, replay_Xc, replay_y = [], [], []

    for i, (label, X, X_clean, y_seg, lr) in enumerate(loaded):
        is_last = i == n_phases - 1
        callbacks = [ens_ckpt, mlflow_cb, live_plot_cb]

        if is_last:
            callbacks = [ens_ckpt, mlflow_cb, live_plot_cb]
            if replay_X:
                n_replay = sum(len(a) for a in replay_y)
                print(f"{label} phase rehearsal: mixing in {n_replay} earlier-phase events "
                      f"(replay_fraction={args.replay_fraction})", flush=True)
                X = np.concatenate([X, *replay_X], axis=0)
                X_clean = np.concatenate([X_clean, *replay_Xc], axis=0)
                y_seg = np.concatenate([y_seg, *replay_y], axis=0)
            early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=False)
            callbacks = [*callbacks, early_stop]
        else:
            rX, rXc, ry = _sample_replay(X, X_clean, y_seg, args.replay_fraction, replay_rng)
            if rX is not None:
                replay_X.append(rX)
                replay_Xc.append(rXc)
                replay_y.append(ry)

        print(f"\n=== Phase: {label} ({len(y_seg)} events, epochs {bounds[i]}->{bounds[i + 1]}) ===", flush=True)
        if i > 0:
            cur_lr = float(ensemble.optimizer.learning_rate.numpy())
            target_lr = min(lr, cur_lr * 3.0)
            print(f"LR -> {target_lr:.2e} (target {lr:.2e})", flush=True)
            ensemble.optimizer.learning_rate.assign(target_lr)

        lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=args.factor, patience=args.lr_patience, min_lr=1e-6)
        hist = _fit(X, X_clean, y_seg, bounds[i], bounds[i + 1], [lr_sched, *callbacks])
        _all_histories.append(hist.history)
        del X, X_clean, y_seg
        gc.collect()

    for i, m in enumerate(sub_models):
        path = os.path.join(args.output_dir, f"ensemble_fno_model_{i}.keras")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble of independent single-pair models with an FNO denoiser.")
    parser.add_argument("train_root_file_low", type=str)
    parser.add_argument("val_root_file", type=str)
    parser.add_argument("--train_root_file_med", type=str, default=None)
    parser.add_argument("--train_root_file_high", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=os.path.join(_HERE, "checkpoints", "ensemble_fno"))
    parser.add_argument("--n_models", type=int, default=3)
    parser.add_argument("--diversity_lambda", type=float, default=0.05)
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
    parser.add_argument("--base", type=int, default=32, help="Segmenter U-Net++ base channels.")
    parser.add_argument("--fno_base", type=int, default=32, help="FNO denoiser hidden channel width.")
    parser.add_argument("--fno_depth", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=20.0)
    parser.add_argument("--low_ratio", type=float, default=0.5)
    parser.add_argument("--med_ratio", type=float, default=0.8)
    parser.add_argument("--replay_fraction", type=float, default=0.15)
    parser.add_argument("--mlflow_experiment", type=str, default="ensemble_fno_denoiser")
    parser.add_argument("--mlflow_run_name", type=str, default=None)
    args = parser.parse_args()

    train_model(args)
