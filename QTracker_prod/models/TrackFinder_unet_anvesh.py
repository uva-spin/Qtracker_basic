""" Custom U-Net for particle track reconstruction """
import os
import numpy as np
import tensorflow as tf  # still needed for backend ops
import keras
from keras import layers, regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications import ResNet50
from typing import Optional
import argparse
from sklearn.model_selection import KFold
from tqdm.keras import TqdmCallback   # added
import sys
def skim_root_file(input_file, max_events, start=0):
    import ROOT
    import numpy as np
    fin = ROOT.TFile.Open(input_file, "READ")
    if not fin or fin.IsZombie():
        raise IOError(f"Could not open file: {input_file}")

    tree = fin.Get("tree")
    if not tree:
        raise KeyError("TTree 'tree' not found in the input file.")

    num_detectors = 62
    num_elementIDs = 201

    X = []
    y_muPlus = []
    y_muMinus = []

    n_entries = min(tree.GetEntries() - start, max_events)
    end = start + n_entries
    for i in range(start, end):
        tree.GetEntry(i)
        event_hits_matrix = np.zeros((num_detectors, num_elementIDs), dtype=np.float32)
        det_ids = getattr(tree, 'detectorID')
        elem_ids = getattr(tree, 'elementID')
        for det_id, elem_id in zip(det_ids, elem_ids):
            if 0 <= det_id < num_detectors and 0 <= elem_id < num_elementIDs:
                event_hits_matrix[det_id, elem_id] = 1
        mu_plus_array = np.array(getattr(tree, 'HitArray_mup'), dtype=np.int32)
        mu_minus_array = np.array(getattr(tree, 'HitArray_mum'), dtype=np.int32)
        X.append(event_hits_matrix)
        y_muPlus.append(mu_plus_array)
        y_muMinus.append(mu_minus_array)

    fin.Close()
    X = np.array(X)[..., np.newaxis]  # Shape: (num_events, 62, 201, 1)
    y_muPlus = np.array(y_muPlus)
    y_muMinus = np.array(y_muMinus)
    print(f"Skimmed {n_entries} events from '{input_file}'")
    return X, y_muPlus, y_muMinus

# Device selection for TensorFlow
def set_tf_device():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Using GPU:", physical_devices[0])
        except Exception as e:
            print("Could not set GPU memory growth:", e)
    elif hasattr(tf.config, 'list_physical_devices') and tf.config.list_physical_devices('MPS'):
        print("Using Apple Silicon MPS")
        # TensorFlow will use MPS automatically if available
    else:
        print("Using CPU")
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('MPS'))
try:
    import wandb
    _WANDB = True
    print('Using W&B')
except ImportError:
    _WANDB = False
    print("Not Using W&B")

from wandb.integration.keras import WandbCallback
_WANDB = False
set_tf_device()

from data_loader import load_data
from losses import custom_loss, cross_validation_loss
# Ensure the checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)


class UNetModel:
    """
    Builds a U-Net-like model for (det, element) grids with 2-class output (mu+, mu-).
    Responsible ONLY for architecture; training/compiling is done by Trainer.
    """

    def __init__(
        self,
        num_detectors: int = 62,
        num_elementIDs: int = 201,
        use_bn: bool = False,
        dropout_bn: float = 0.0,
        dropout_enc: float = 0.0,
        backbone: Optional[str] = None,
        l2_reg: float = 1e-4,
        backbone_weights: Optional[str] = None,
    ):
        self.num_detectors = num_detectors
        self.num_elementIDs = num_elementIDs
        self.use_bn = use_bn
        self.dropout_bn = dropout_bn
        self.dropout_enc = dropout_enc
        self.backbone = backbone
        self.l2_reg = l2_reg
        self.backbone_weights = backbone_weights

    @staticmethod
    def crop_to_match(skip, up):
        """Crop skip connection to match the size of upsampled tensor."""
        def crop(inputs):
            skip_tensor, up_tensor = inputs
            sh, sw = tf.shape(skip_tensor)[1], tf.shape(skip_tensor)[2]
            uh, uw = tf.shape(up_tensor)[1], tf.shape(up_tensor)[2]
            crop_h = sh - uh
            crop_w = sw - uw
            crop_top = crop_h // 2
            crop_bottom = crop_h - crop_top
            crop_left = crop_w // 2
            crop_right = crop_w - crop_left
            return skip_tensor[:, crop_top:sh - crop_bottom, crop_left:sw - crop_right, :]
        return layers.Lambda(crop)([skip, up])

    def build(self):
        inputs = keras.Input(shape=(self.num_detectors, self.num_elementIDs, 1))

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        b = layers.Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)

        # Decoder with cropping
        u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(b)
        c4_crop = UNetModel.crop_to_match(c4, u4)
        u4 = layers.Concatenate()([u4, c4_crop])

        u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(u4)
        c3_crop = UNetModel.crop_to_match(c3, u3)
        u3 = layers.Concatenate()([u3, c3_crop])

        u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(u3)
        c2_crop = UNetModel.crop_to_match(c2, u2)
        u2 = layers.Concatenate()([u2, c2_crop])

        u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u2)
        c1_crop = UNetModel.crop_to_match(c1, u1)
        u1 = layers.Concatenate()([u1, c1_crop])

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)

        # Dense classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # Output shape: (2, num_detectors, num_elementIDs)
        x = layers.Dense(2 * self.num_detectors * self.num_elementIDs, activation="softmax")(x)
        outputs = layers.Reshape((2, self.num_detectors, self.num_elementIDs))(x)

        return keras.Model(inputs, outputs, name="UNet")


def build_model(
    num_detectors: int = 62,
    num_elementIDs: int = 201,
    use_bn: bool = False,
    dropout_bn: float = 0.0,
    dropout_enc: float = 0.0,
    backbone: Optional[str] = None,
    l2_reg: float = 1e-4,
    backbone_weights: Optional[str] = None,
) -> tf.keras.Model:
    return UNetModel(
        num_detectors=num_detectors,
        num_elementIDs=num_elementIDs,
        use_bn=use_bn,
        dropout_bn=dropout_bn,
        dropout_enc=dropout_enc,
        backbone=backbone,
        l2_reg=l2_reg,
        backbone_weights=backbone_weights,
    ).build()


def train_model(
    train_root_file,
    val_root_file,
    output_model,
    learning_rate=0.00005,
    patience=5,
    use_bn=False,
    dropout_bn=0.0,
    dropout_enc=0.0,
    backbone=None,
    k_folds=None,
    # --- W&B options ---
    use_wandb=True,
    wandb_project="my-project",
    wandb_entity=None,
    wandb_run_name=None,
    wandb_group=None,
    wandb_tags=None,
    wandb_mode=None,              # "online" | "offline" | "disabled"
    extra_wandb_config=None,
    # --- training options ---
    epochs=40,
    batch_size=8,
    random_state=42,
    skim = False,
    skim_max_events = 1000,
):
    import matplotlib.pyplot as plt
    from tqdm.keras import TqdmCallback

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=max(1, patience // 3),
        min_lr=1e-7
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # W&B init (optional)
    run = None
    if use_wandb and _WANDB:
        cfg = dict(
            learning_rate=learning_rate, patience=patience,
            use_bn=use_bn, dropout_bn=dropout_bn, dropout_enc=dropout_enc,
            backbone=backbone, k_folds=k_folds, epochs=epochs, batch_size=batch_size
        )
        if extra_wandb_config:
            cfg.update(extra_wandb_config)
        run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_run_name,
            group=wandb_group, tags=wandb_tags, mode=wandb_mode, config=cfg
        )

    # Data
    if skim:
        X_train, y_muPlus_train, y_muMinus_train = skim_root_file(train_root_file, skim_max_events, start=0)
        if train_root_file != val_root_file:
            X_val, y_muPlus_val, y_muMinus_val = skim_root_file(val_root_file, skim_max_events, start=0)
        else:
            X_val, y_muPlus_val, y_muMinus_val = None, None, None
    else:
        X_train, y_muPlus_train, y_muMinus_train = load_data(train_root_file)
        X_val = None
        if train_root_file != val_root_file:
            X_val, y_muPlus_val, y_muMinus_val = load_data(val_root_file)
    if X_train is None or (train_root_file != val_root_file and X_val is None):
        print("No data found.")
        if run:
            run.finish()
        return

    histories = []

    if k_folds:
        # Concatenate and run K-Fold
        if train_root_file != val_root_file:
            X = np.concatenate([X_train, X_val], axis=0)
            y_muPlus = np.concatenate([y_muPlus_train, y_muPlus_val], axis=0)
            y_muMinus = np.concatenate([y_muMinus_train, y_muMinus_val], axis=0)
            y = np.stack([y_muPlus, y_muMinus], axis=1)
        else:
            X = X_train
            y_muPlus = y_muPlus_train
            y_muMinus = y_muMinus_train
            y = np.stack([y_muPlus, y_muMinus], axis=1)

        kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=random_state)
        best_val = np.inf
        best_model_path = os.path.join("checkpoints", "best_of_folds.keras")

        fold_val_losses = []
        fold_models = []

        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            print(f"[KFold] Fold {fold}/{k_folds}")
            model = build_model(
                use_bn=use_bn, dropout_bn=dropout_bn, dropout_enc=dropout_enc, backbone=backbone
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

            cbs = [lr_scheduler, early_stopping, TqdmCallback(verbose=1)]
            if use_wandb and _WANDB:
                cbs.append(WandbCallback(save_model=False))

            history = model.fit(
                X[tr], y[tr],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X[va], y[va]),
                callbacks=cbs,
                verbose=0
            )
            histories.append(history)

            # Predict on validation set for this fold
            y_pred = model.predict(X[va], batch_size=batch_size, verbose=1)
            fold_loss = custom_loss(tf.convert_to_tensor(y[va]), tf.convert_to_tensor(y_pred)).numpy()
            fold_val_losses.append(fold_loss)
            fold_models.append(model)
            print(f"Fold {fold} custom_loss: {fold_loss:.6f}")

        # Compute cross-validation loss using all folds
        y_trues = [tf.convert_to_tensor(y[va]) for _, va in kf.split(X)]
        y_preds = [tf.convert_to_tensor(fold_models[i].predict(X[va], batch_size=batch_size, verbose=1)) for i, (_, va) in enumerate(kf.split(X))]
        cv_loss = cross_validation_loss(y_trues, y_preds, loss_fn=custom_loss).numpy()
        print(f"Cross-validation average loss: {cv_loss:.6f}")

        # Save the model from the fold with the lowest validation loss
        best_fold_idx = int(np.argmin(fold_val_losses))
        fold_models[best_fold_idx].save(best_model_path)

        # Save best-of-folds to requested output path
        tf.keras.models.load_model(best_model_path).save(output_model)
        print(f"Best-of-{k_folds} folds model saved to {output_model}")

    else:
        # Single split training
        y_train = np.stack([y_muPlus_train, y_muMinus_train], axis=1)
        y_val = np.stack([y_muPlus_val, y_muMinus_val], axis=1)

        model = build_model(use_bn=use_bn, dropout_bn=dropout_bn, dropout_enc=dropout_enc, backbone=backbone)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

        cbs = [
            lr_scheduler,
            early_stopping,
            ModelCheckpoint("checkpoints/best_single_split.keras", monitor="val_loss", save_best_only=True),
            TqdmCallback(verbose=1)
        ]
        if use_wandb and _WANDB:
            cbs.append(WandbCallback(save_model=False))

        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=cbs,
            verbose=0
        )
        histories.append(history)

        # Save best checkpoint as final output
        best_path = "checkpoints/best_single_split.keras"
        if os.path.exists(best_path):
            tf.keras.models.load_model(best_path).save(output_model)
        else:
            model.save(output_model)
        print(f"Model saved to {output_model}")

    # Plot training and validation losses
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
    except NameError:
        base_dir = os.getcwd()
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if k_folds:
        all_train = [h.history['loss'] for h in histories]
        all_val = [h.history['val_loss'] for h in histories]
        max_epochs = max(len(l) for l in all_train)
        train_matrix = np.array([np.pad(l, (0, max_epochs - len(l)), constant_values=np.nan) for l in all_train])
        val_matrix = np.array([np.pad(l, (0, max_epochs - len(l)), constant_values=np.nan) for l in all_val])
        mean_train = np.nanmean(train_matrix, axis=0)
        mean_val = np.nanmean(val_matrix, axis=0)
        std_train = np.nanstd(train_matrix, axis=0)
        std_val = np.nanstd(val_matrix, axis=0)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_train, label='Train Loss (mean)')
        plt.fill_between(range(max_epochs), mean_train - std_train, mean_train + std_train, alpha=0.2)
        plt.plot(mean_val, label='Val Loss (mean)')
        plt.fill_between(range(max_epochs), mean_val - std_val, mean_val + std_val, alpha=0.2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Cross-validation Loss Curves ({k_folds} folds)")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "cv_losses.png"), bbox_inches='tight')
        plt.close()
    else:
        history = histories[0]
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Val Loss Over Epochs')
        plt.savefig(os.path.join(plot_dir, "losses.png"), bbox_inches='tight')
        plt.close()

    if run:
        run.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("--train_root_file",default="/Users/anvesh/Documents/research_position/TrackFinder Files/mc_events_train.root" ,type=str, help="Path to the train ROOT file.")
    parser.add_argument("--val_root_file", default = '/Users/anvesh/Documents/research_position/TrackFinder Files/mc_events_train.root', type=str, help="Path to the validation ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for EarlyStopping.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--dropout_bn", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    parser.add_argument("--dropout_enc", type=float, default=0.0, help="Dropout rate for encoder blocks.")
    parser.add_argument("--backbone", type=str, default='resnet50', help="Backbone encoder. Available: [None, 'resnet50'].")
    parser.add_argument("--k_folds", type=str, default=10, help="Number of folds for Cross validation. If not set cross-validation will not be used")
    # W&B
    parser.add_argument("--use_wandb", type=int, default=0, help="Enable Weights & Biases logging: [0 = False, 1 = True].")
    parser.add_argument("--wandb_project", type=str, default="WandB-Test", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (user or team).")
    parser.add_argument("--wandb_run_name", type=str, default='test_run', help="W&B run name.")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group (useful for k-folds).")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags, e.g. 'unet,bn,resnet50'.")
    parser.add_argument("--wandb_mode", type=str, default=None, help="W&B mode: 'online', 'offline', or 'disabled'.")

    args = parser.parse_args()

    use_bn = bool(args.batch_norm)
    k_folds = args.k_folds if (args.k_folds is not None and args.k_folds >= 2) else None
    use_wandb = bool(args.use_wandb)
    wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None

    train_model(
        args.train_root_file,
        args.val_root_file,
        args.output_model,
        args.learning_rate,
        patience=args.patience,
        use_bn=use_bn,
        dropout_bn=args.dropout_bn,
        dropout_enc=args.dropout_enc,
        backbone=args.backbone,
        k_folds=k_folds,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=wandb_tags,
        wandb_mode=args.wandb_mode,
    )
