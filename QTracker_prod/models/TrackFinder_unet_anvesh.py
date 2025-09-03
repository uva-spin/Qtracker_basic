""" Custom U-Net for particle track reconstruction """

import os
import ROOT
import numpy as np
import tensorflow as tf
import argparse
import math
import sklearn
from sklearn.model_selection import Kfold
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

from data_loader import load_data,
from losses import custom_loss
import wandb
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


def build_model(self) -> tf.keras.Model:
    det, elem = self.num_detectors, self.num_elementIDs
    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    n = 5 if backbone == 'resnet50' else 4
    num_pool = 2 ** n   # 2 ^ n, n = number of max pooling
    closest_even_det = num_pool * math.ceil(num_detectors / num_pool)
    closest_even_elem = num_pool * math.ceil(num_elementIDs / num_pool)
    det_diff = closest_even_det - num_detectors
    elem_diff = closest_even_elem - num_elementIDs
    padding = (
        (det_diff // 2, det_diff - det_diff // 2),
        (elem_diff // 2, elem_diff - elem_diff // 2)
    )

    x = layers.ZeroPadding2D(padding=padding)(input_layer)

    # Encoder
    if backbone == 'resnet50':
        x = layers.Concatenate()([x, x, x])
        # x = Conv2D(3, kernel_size=1, padding='same')(x)
        backbone = ResNet50(include_top=False, input_tensor=x)

        # Partially freeze backbone and alter stride for compatibility with U-Net decoder
        backbone.trainable = False
        for layer in backbone.layers:
            if layer.name == 'conv1_conv':
                layer.strides = (1, 1)
            if layer.name.startswith('conv5_') or layer.name.startswith('conv4_'):
                layer.trainable = True

        enc1 = backbone.get_layer('conv1_relu').output
        enc2 = backbone.get_layer('conv2_block3_out').output
        enc3 = backbone.get_layer('conv3_block4_out').output
        enc4 = backbone.get_layer('conv4_block6_out').output
        enc5 = backbone.get_layer('conv5_block3_out').output
    else:
        enc1 = unet_block(x, 64, use_bn=use_bn)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)

        enc2 = unet_block(pool1, 128, use_bn=use_bn, dropout_enc=dropout_enc)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)

        enc3 = unet_block(pool2, 256, use_bn=use_bn)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)

        enc4 = unet_block(pool3, 512, use_bn=use_bn, dropout_enc=dropout_enc)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)

        # Bottleneck
        enc5 = unet_block(pool4, 1024, use_bn=use_bn, dropout_bn=dropout_bn)

    # Decoder
    dec1 = layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding='same')(enc5) # padding same avoids cropping
    dec1 = layers.concatenate([dec1, enc4])    # skip connections
    dec1 = unet_block(dec1, 512, use_bn=use_bn)

    dec2 = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(dec1) # padding same avoids cropping
    dec2 = layers.concatenate([dec2, enc3])    # skip connections
    dec2 = unet_block(dec2, 256, use_bn=use_bn)

    dec3 = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(dec2) # padding same avoids cropping
    dec3 = layers.concatenate([dec3, enc2])    # skip connections
    dec3 = unet_block(dec3, 128, use_bn=use_bn)

    dec4 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(dec3)
    dec4 = layers.concatenate([dec4, enc1])    # skip connections
    dec4 = layers.Cropping2D(cropping=padding)(dec4)   # Remove extra padding
    dec4 = unet_block(dec4, 64, use_bn=use_bn)

    # Output layer
    x = layers.Conv2D(2, kernel_size=1)(dec4)
    x = layers.Softmax(axis=2)(x)      # softmax over elementID
    output = layers.Permute((3, 1, 2))(x)  # permute to match shape required by loss: (batch, 2, det, elem)

    # Initialize model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def train_model(
    train_root_file,
    val_root_file,
    output_model,
    learning_rate=0.00005,
    patience=5,
    use_bn=False,
    dropout_bn=0.0,
    dropout_enc=0.0,
    backbone=None, k_folds = None
    use_wandb=True,
    wandb_project="my-project",
    wandb_entity=None,            # set if you use a W&B team/org
    wandb_run_name=None,
    wandb_group=None,             # helpful when using k-folds
    wandb_tags=None,              # e.g., ["resnet50", "bn"]
    wandb_mode=None,              # "online" | "offline" | "disabled"
    extra_wandb_config=None,      # dict of extra config entries
    # --- training options ---
    epochs=40,
    batch_size=32,
    random_state=42
):
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=patience // 3,
        min_lr=1e-7
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    model = build_model(use_bn=use_bn, dropout_bn=dropout_bn, dropout_enc=dropout_enc, backbone=backbone)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['accuracy']
    )

    if k_folds:
        X_train, y_muPlus_train, y_muMinus_train = load_data(train_root_file)
        X_val, y_muPlus_val, y_muMinus_val = load_data(val_root_file)
        if X_train is  None or  X_val is None:
            return
        X = np.concatenate([X_train, X_val], axis=0)
        y_muPlus = np.concatenate([y_muPlus_train, y_muPlus_val], axis=0)
        y_muMinus = np.concatenate([y_muMinus_train, y_muMinus_val], axis=0)
        y = np.stack([y_muPlus, y_muMinus], axis=1)  # Shape: (num_events, 2, 62)


    else:
        X_train, y_muPlus_train, y_muMinus_train = load_data(train_root_file)
        if X_train is None:
            return
        y_train = np.stack([y_muPlus_train, y_muMinus_train], axis=1)  # Shape: (num_events, 2, 62)

        X_val, y_muPlus_val, y_muMinus_val = load_data(val_root_file)
        if X_val is None:
            return
        y_val = np.stack([y_muPlus_val, y_muMinus_val], axis=1)  # Shape: (num_events, 2, 62)


    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping])

    # Plot train and val loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Val Loss Over Epochs')

    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "losses.png"))
    plt.show()

    model.save(output_model)
    print(f"Model saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TensorFlow model to predict hit arrays from event hits.")
    parser.add_argument("train_root_file", type=str, help="Path to the train ROOT file.")
    parser.add_argument("val_root_file", type=str, help="Path to the validation ROOT file.")
    parser.add_argument("--output_model", type=str, default="checkpoints/track_finder.h5", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for EarlyStopping.")
    parser.add_argument("--batch_norm", type=int, default=0, help="Flag to set batch normalization: [0 = False, 1 = True].")
    parser.add_argument("--dropout_bn", type=float, default=0.0, help="Dropout rate for bottleneck layer.")
    parser.add_argument("--dropout_enc", type=float, default=0.0, help="Dropout rate for encoder blocks.")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone encoder. Available: [None, 'resnet50'].")
    parser.add_argument("--k_folds", type=str, default=None, help="Number of folds for Cross validation. If not set cross-validation will not be used")
    args = parser.parse_args()

    use_bn = bool(args.batch_norm)

    train_model(
        args.train_root_file,
        args.val_root_file,
        args.output_model,
        args.learning_rate,
        patience=args.patience,
        use_bn=use_bn,
        dropout_bn=args.dropout_bn,      # recommend 0.5 as starting point,
        dropout_enc=args.dropout_enc,    # recommend 0.1-0.3 as starting point
        backbone=args.backbone
    )
