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

try:
    import wandb
    from wandb.integration.keras import WandbCallback
    _WANDB = True
    print('Using W&B')
except ImportError:
    _WANDB = False
    print("Not Using W&B")

set_tf_device()

from data_loader import load_data
from losses import custom_loss
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
        backbone: Optional[str] = None,  # "resnet50" or None
        l2_reg: float = 1e-4,
        backbone_weights: Optional[str] = None,  # "imagenet" or None
    ):
        self.num_detectors = num_detectors
        self.num_elementIDs = num_elementIDs
        self.use_bn = use_bn
        self.dropout_bn = dropout_bn
        self.dropout_enc = dropout_enc
        self.backbone = backbone
        self.l2_reg = l2_reg
        self.backbone_weights = backbone_weights

    def unet_block(x, filters, l2=1e-4, use_bn=False, dropout_bn=0.0, dropout_enc=0.0):
    # First Conv Layer + Activation
        x = layers.Conv2D(
            filters, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.l2(l2)
    )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Dropout for bottleneck layers
        if dropout_bn > 0:
            x = layers.Dropout(dropout_bn)(x)

    # Second Conv Layer
        x = layers.Conv2D(
            filters, kernel_size=3, padding='same',
            kernel_regularizer=regularizers.l2(l2)
    )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Dropout for encoder blocks
        if dropout_enc > 0:
            x = layers.Dropout(dropout_enc)(x)
        return x
    
    def build(self):
        inputs = keras.Input(shape=(self.num_detectors, self.num_elementIDs, 1))

        # Encoder
        c1 = UNetModel.unet_block(inputs, 32, l2=self.l2_reg,
                                  use_bn=self.use_bn,
                                  dropout_bn=self.dropout_bn,
                                  dropout_enc=self.dropout_enc)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = UNetModel.unet_block(p1, 64, l2=self.l2_reg,
                                  use_bn=self.use_bn,
                                  dropout_bn=self.dropout_bn,
                                  dropout_enc=self.dropout_enc)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        # Bottleneck
        bn = UNetModel.unet_block(p2, 128, l2=self.l2_reg,
                                  use_bn=self.use_bn,
                                  dropout_bn=self.dropout_bn,
                                  dropout_enc=self.dropout_enc)

        # Decoder
        u1 = layers.UpSampling2D((2, 2))(bn)

        # --- resize c2 to match u1 ---
        c2 = layers.Resizing(height=u1.shape[1], width=u1.shape[2])(c2)

        u1 = layers.Concatenate()([u1, c2])
        c3 = UNetModel.unet_block(u1, 64, l2=self.l2_reg,
                                  use_bn=self.use_bn,
                                  dropout_bn=self.dropout_bn,
                                  dropout_enc=self.dropout_enc)

        u2 = layers.UpSampling2D((2, 2))(c3)

        # --- resize c1 to match u2 ---
        c1 = layers.Resizing(height=u2.shape[1], width=u2.shape[2])(c1)

        u2 = layers.Concatenate()([u2, c1])
        c4 = UNetModel.unet_block(u2, 32, l2=self.l2_reg,
                                  use_bn=self.use_bn,
                                  dropout_bn=self.dropout_bn,
                                  dropout_enc=self.dropout_enc)

        # Output (2-class softmax)
        outputs = layers.Conv2D(2, (1, 1), activation="softmax")(c4)

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
    batch_size=32,
    random_state=42
):
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
        if extra_wandb_config: cfg.update(extra_wandb_config)
        run =  wandb.init(
             project=wandb_project, entity=wandb_entity, name=wandb_run_name,
             group=wandb_group, tags=wandb_tags, mode=wandb_mode, config=cfg
         )

    # Data
    X_train, y_muPlus_train, y_muMinus_train = load_data(train_root_file)
    X_val, y_muPlus_val, y_muMinus_val = load_data(val_root_file)
    if X_train is None or X_val is None:
        print("No data found.")
        if run: run.finish()
        return

    if k_folds:
        # Concatenate and run K-Fold
        X = np.concatenate([X_train, X_val], axis=0)
        y_muPlus = np.concatenate([y_muPlus_train, y_muPlus_val], axis=0)
        y_muMinus = np.concatenate([y_muMinus_train, y_muMinus_val], axis=0)
        y = np.stack([y_muPlus, y_muMinus], axis=1)

        kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=random_state)
        best_val = np.inf
        best_model_path = os.path.join("checkpoints", "best_of_folds.keras")

        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            print(f"[KFold] Fold {fold}/{k_folds}")
            model = build_model(
                use_bn=use_bn, dropout_bn=dropout_bn, dropout_enc=dropout_enc, backbone=backbone
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])

            cbs = [lr_scheduler, early_stopping]
            if use_wandb and _WANDB:
                cbs.append(WandbCallback(save_model=False))

            history = model.fit(
                X[tr], y[tr],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X[va], y[va]),
                callbacks=cbs,
                verbose=1
            )
            fold_best = float(np.min(history.history.get("val_loss", [np.inf])))
            print(f"Fold {fold} best val_loss: {fold_best:.6f}")
            if fold_best < best_val:
                best_val = fold_best
                model.save(best_model_path)

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

        cbs = [lr_scheduler, early_stopping, ModelCheckpoint("checkpoints/best_single_split.keras", monitor="val_loss", save_best_only=True)]
        if use_wandb and _WANDB:
            cbs.append(WandbCallback(save_model=False))

        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=cbs,
            verbose=1
        )

        # Plot train and val loss over epochs
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
        except NameError:
            base_dir = os.getcwd()
        plot_dir = os.path.join(base_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        plt.title('Training and Val Loss Over Epochs')
        plt.savefig(os.path.join(plot_dir, "losses.png"), bbox_inches='tight')
        plt.close()

        # Save best checkpoint as final output
        best_path = "checkpoints/best_single_split.keras"
        if os.path.exists(best_path):
            tf.keras.models.load_model(best_path).save(output_model)
        else:
            model.save(output_model)
        print(f"Model saved to {output_model}")

    if run:
        run.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("--train_root_file",default="/Users/anvesh/Documents/research_position/TrackFinder Files/mc_events_train.root" ,type=str, help="Path to the train ROOT file.")
    parser.add_argument("--val_root_file", default = '/Users/anvesh/Documents/research_position/TrackFinder Files/mc_events_train.root', type=str, help="Path to the validation ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for EarlyStopping.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--dropout_bn", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    parser.add_argument("--dropout_enc", type=float, default=0.0, help="Dropout rate for encoder blocks.")
    parser.add_argument("--backbone", type=str, default='resnet50', help="Backbone encoder. Available: [None, 'resnet50'].")
    parser.add_argument("--k_folds", type=str, default=10, help="Number of folds for Cross validation. If not set cross-validation will not be used")
    # W&B
    parser.add_argument("--use_wandb", type=int, default=1, help="Enable Weights & Biases logging: [0 = False, 1 = True].")
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
