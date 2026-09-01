"""
FNO-based denoiser: alternates FourierBlock1D (spectral mixing along the
elementID axis) with AxialAttention (mixing across the categorical
detectorID axis), operating at full (62, 201) resolution throughout --
no encoder/decoder, no up/downsampling. Denoising is inherently a
full-resolution per-pixel task, and neither block ever changes spatial
shape, so there's nothing for a U-Net-style bottleneck to buy here that
the classifier's backbone (models/experiments/pair_count_fno) didn't
already validate the individual blocks on.

Output is a single-channel logit map matching MultiTrackFinder.py's
denoise head convention (linear, no activation) -- trained with
models/losses.py's weighted_bce(pos_weight=...), which expects logits.
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


def build_denoiser(
    num_detectors: int = NUM_DETECTORS,
    num_elementIDs: int = NUM_ELEMENT_IDS,
    base: int = 32,
    fno_depth: int = 4,
    k_max: int = 32,
    num_heads: int = 4,
) -> tf.keras.Model:
    """
    Args:
        num_detectors: Number of detector layers (default 62).
        num_elementIDs: Number of element IDs per layer (default 201).
        base: Hidden channel width used throughout the FNO/attention stack.
        fno_depth: Number of (FourierBlock1D, AxialAttention) pairs.
        k_max: Number of low-frequency Fourier modes kept per block.
        num_heads: Attention heads for the detector-axis mixing.

    Returns:
        tf.keras.Model mapping (batch, 62, 201, 1) -> (batch, 62, 201, 1) logits.
    """
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

    denoise_out = layers.Conv2D(1, kernel_size=1, name="denoise", dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=denoise_out, name="fno_denoiser")
    return model
