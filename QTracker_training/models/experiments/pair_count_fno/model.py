"""
Axial-FNO pair-count classifier.

Predicts the number of dimuon pairs (0..max_pairs) per event from the noisy
hit matrix. Two-stage-pipeline "Stage A" candidate: an encoder-only model
(no decoder -- classification only needs a pooled representation) that
alternates two forms of global mixing:

  - FourierBlock1D along elementID: elementID is a genuine physical
    coordinate (wire/channel position), so a spectral conv gives a global
    receptive field along it at O(N log N) cost, cheaper than attention.
  - AxialAttention along detectorID (height axis, no FFN): detectorID is
    categorical (station/plane index, not a continuous coordinate), so it's
    mixed with the existing proven attention layer instead of FFT.
"""

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from fno_layers import FourierBlock1D  # noqa: E402
from layers import AxialAttention  # noqa: E402

NUM_DETECTORS = 62
NUM_ELEMENT_IDS = 201


def build_model(
    num_detectors: int = NUM_DETECTORS,
    num_elementIDs: int = NUM_ELEMENT_IDS,
    max_pairs: int = 3,
    base: int = 32,
    fno_depth: int = 4,
    k_max: int = 32,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    Args:
        num_detectors: Number of detector layers (default 62).
        num_elementIDs: Number of element IDs per layer (default 201).
        max_pairs: Maximum number of dimuon pairs the model must count;
            output classes are 0..max_pairs (max_pairs + 1 classes).
        base: Hidden channel width used throughout the FNO/attention stack.
        fno_depth: Number of (FourierBlock1D, AxialAttention) pairs.
        k_max: Number of low-frequency Fourier modes kept per block.
        num_heads: Attention heads for the detector-axis mixing.
        dropout: Dropout rate before the final classification head.

    Returns:
        tf.keras.Model mapping (batch, 62, 201, 1) -> (batch, max_pairs + 1) softmax.
    """
    num_classes = max_pairs + 1
    input_layer = layers.Input(shape=(num_detectors, num_elementIDs, 1))
    x = layers.Conv2D(base, kernel_size=1, name="lift")(input_layer)

    for i in range(fno_depth):
        x = FourierBlock1D(channels=base, k_max=k_max, name=f"fourier_block_{i}")(x)
        x = AxialAttention(
            embed_dim=base,
            num_heads=num_heads,
            axis="height",
            use_ffn=False,
            name=f"detector_mix_{i}",
        )(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(
        num_classes, activation="softmax", dtype=tf.float32, name="pair_count"
    )(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output, name="axial_fno_pair_classifier")
    return model
