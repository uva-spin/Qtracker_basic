from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="pair_count_fno", name="FourierBlock1D")
class FourierBlock1D(layers.Layer):
    """
    1D Fourier Neural Operator block (Li et al., 2021) applied along the
    elementID axis of a (batch, detector, elementID, channels) tensor.

    elementID indexes a real physical coordinate (wire/channel position
    along a detector plane), so a spectral convolution along that axis
    gives a global receptive field at O(N log N) cost -- the FNO learns a
    per-frequency channel mixing on the truncated low-frequency modes,
    combined with a pointwise linear skip (the standard FNO block).
    detectorID is categorical (station/plane index), not a continuous
    coordinate, so it is intentionally left untouched here -- mix it with
    AxialAttention(axis="height") instead (see model.py).
    """

    def __init__(
        self,
        channels: int,
        k_max: int = 32,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            channels (int): Number of input/output channels (must match).
            k_max (int): Number of low-frequency Fourier modes to keep.
            activation (str): Activation applied after the skip connection.
        """
        super(FourierBlock1D, self).__init__(**kwargs)
        self.channels = channels
        self.k_max = k_max
        self.activation_name = activation
        self.activation = layers.Activation(activation)
        self.skip = layers.Conv2D(channels, kernel_size=1, dtype=tf.float32)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Builds the learned complex spectral weight (real/imag parts, since
        Keras weights must be real-valued)."""
        _, _, _, C = input_shape
        if C != self.channels:
            raise ValueError(
                f"FourierBlock1D expects {self.channels} input channels, got {C}."
            )
        scale = 1.0 / (self.channels * self.k_max) ** 0.5
        init = tf.keras.initializers.RandomNormal(stddev=scale)
        self.w_real = self.add_weight(
            shape=(self.channels, self.channels, self.k_max),
            initializer=init,
            trainable=True,
            dtype=tf.float32,
            name="spectral_weight_real",
        )
        self.w_imag = self.add_weight(
            shape=(self.channels, self.channels, self.k_max),
            initializer=init,
            trainable=True,
            dtype=tf.float32,
            name="spectral_weight_imag",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x (tf.Tensor): Input tensor of shape (B, D, L, C).

        Returns:
            tf.Tensor: Output tensor of shape (B, D, L, C).
        """
        x32 = tf.cast(x, tf.float32)
        skip = self.skip(x32)

        L = x32.shape[2]
        x_t = tf.transpose(x32, perm=[0, 1, 3, 2])  # (B, D, C, L)
        x_ft = tf.signal.rfft(x_t)  # (B, D, C, L // 2 + 1), complex64

        freq_len = x_ft.shape[-1]
        k = min(self.k_max, freq_len)
        x_ft_trunc = x_ft[..., :k]  # (B, D, C_in, k)

        weight = tf.complex(self.w_real[:, :, :k], self.w_imag[:, :, :k])  # (C_in, C_out, k)
        out_ft_trunc = tf.einsum("bdik,iok->bdok", x_ft_trunc, weight)  # (B, D, C_out, k)

        out_ft = tf.pad(out_ft_trunc, [[0, 0], [0, 0], [0, 0], [0, freq_len - k]])
        out = tf.signal.irfft(out_ft, fft_length=[L])  # (B, D, C_out, L)
        out = tf.transpose(out, perm=[0, 1, 3, 2])  # (B, D, L, C_out)

        out = self.activation(out + skip)
        return tf.cast(out, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "k_max": self.k_max,
            "activation": self.activation_name,
        })
        return config
