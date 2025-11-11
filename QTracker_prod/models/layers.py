import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from typing import Any


def conv_block(
    x: tf.Tensor, 
    filters: int, 
    l2: float = 1e-4, 
    use_bn: bool = False, 
    dropout_bn: float = 0.0, 
    dropout: float = 0.0,
) -> tf.Tensor:
    """
    A convolutional block with two Conv2D layers, optional batch normalization,
    ReLU activations, and dropout. Includes a residual connection.
    Used in U-Net architectures.
    
    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the Conv2D layers.
        l2 (float): L2 regularization factor.
        use_bn (bool): Whether to use batch normalization.
        dropout_bn (float): Dropout rate after the first Conv2D layer (for bottleneck layers).
        dropout (float): Dropout rate after the second Conv2D layer (for encoder blocks).
    
    Returns:
        tf.Tensor: Output tensor after applying the convolutional block.
    """

    shortcut = x

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

    # Project shortcut if needed
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    
    x = layers.Activation('relu')(x)

    # Dropout for encoder blocks
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def upsample(x: tf.Tensor) -> tf.Tensor:
    """
    Upsamples the input tensor by a factor of 2 using bilinear interpolation.
    
    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Upsampled tensor.
    """

    x = layers.UpSampling2D(interpolation="bilinear")(x)
    return x


class AxialAttention(layers.Layer):
    """
    Axial Attention Layer as described in "Axial Attention in Multidimensional Transformers"
    (Ho et al., 2019). This layer applies self-attention along a specified axis (height or width)
    of the input tensor, followed by an optional feed-forward network (FFN).
    """

    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        axis: str = 'height', 
        dropout: float = 0.0, 
        use_ffn: bool = True, 
        **kwargs: Any,
    ) -> None:
        """
        Initializes the AxialAttention layer.

        Args:
            embed_dim (int): Dimensionality of the embedding.
            num_heads (int): Number of attention heads.
            axis (str): Axis to apply attention ('height' or 'width').
            dropout (float): Dropout rate.
            use_ffn (bool): Whether to include the feed-forward network.
            **kwargs: Additional keyword arguments for the base Layer class.
        """

        super(AxialAttention, self).__init__(**kwargs)

        self.axis = axis
        self.use_ffn = use_ffn

        self.lnorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.lnorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.add = layers.Add()
        self.dropout = layers.Dropout(dropout)

        if use_ffn:
            # Convolutional Feed-Forward Network
            self.ffn = tf.keras.Sequential([
                layers.Conv2D(embed_dim * 4, kernel_size=1, activation='gelu'),
                layers.DepthwiseConv2D(kernel_size=3, padding='same'),  # for spatial mixing
                layers.Dropout(dropout),
                layers.Conv2D(embed_dim, kernel_size=1),
                layers.Dropout(dropout),
            ])

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the learned absolute positional encoding for the specified axis. 
        This method is called when the layer is first used.
        """

        _, D, E, C = input_shape
        L = D if self.axis == 'height' else E

        self.pos_enc = self.add_weight(
            shape=(L, C),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name=f'pos_enc_{self.axis}',
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies the Axial Attention layer to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (B, D, E, C).
        
        Returns:
            tf.Tensor: Output tensor after applying axial attention and optional FFN.
        """

        B, D, E, C = tf.unstack(tf.shape(x))

        # Apply positional encoding and reshape for attention
        if self.axis == 'height':
            pos_enc = tf.reshape(self.pos_enc, (1, D, 1, C))
            x = x + pos_enc
            x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, E, D, C)
            x = tf.reshape(x, (B * E, D, C))        # (B * E, D, C)
        else:
            pos_enc = tf.reshape(self.pos_enc, (1, 1, E, C))
            x = x + pos_enc
            x = tf.reshape(x, (B * D, E, C))        # (B * D, E, C)

        # Layer norm + Multi-head Self-Attention
        skip = x
        x = self.lnorm1(x)
        x = self.attention(x, x)  # Self-attention
        x = self.dropout(x)
        x = tf.cast(x, skip.dtype)      # Ensure dtype consistency
        x = self.add([x, skip])  # Residual connection

        # Restore original shape
        if self.axis == 'height':
            x = tf.reshape(x, (B, E, D, C))
            x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, D, E, C)
        else:
            x = tf.reshape(x, (B, D, E, C))

        # Feed-forward network with residual connection
        if self.use_ffn:
            skip = x
            x = self.ffn(self.lnorm2(x))
            x = tf.cast(x, skip.dtype)      # Ensure dtype consistency
            x = self.add([x, skip])

        return x
