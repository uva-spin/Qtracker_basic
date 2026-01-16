import math
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from typing import Optional

from layers import conv_block, upsample, AxialAttention


def unet_backbone(
    input_layer: layers.Input,
    num_detectors: int,
    num_elementIDs: int,
    use_bn: bool,
    dropout_bn: float,
    dropout_enc: float,
    base: int,
    backbone: Optional[str] = None,
) -> layers.Layer:
    """
    U-Net backbone with optional ResNet50 encoder.

    Args:
        input_layer (layers.Input): Input layer to the backbone.
        num_detectors (int): Number of detector channels in the input.
        num_elementIDs (int): Number of element ID channels in the input.
        use_bn (bool): Whether to use batch normalization.
        dropout_bn (float): Dropout rate after bottleneck if using batch normalization.
        dropout_enc (float): Dropout rate in the encoder.
        base (int): Base number of filters for the U-Net.
        backbone (Optional[str]): Backbone type, either 'resnet50' or None for standard U-Net.

    Returns:
        layers.Layer: Output layer of the U-Net backbone.
    """

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    filters = [base, base * 2, base * 4, base * 8, base * 16]
    n = 5 if backbone == "resnet50" else len(filters) - 1
    num_pool = 2**n  # 2 ^ n, n = number of max pooling
    closest_even_det = num_pool * math.ceil(num_detectors / num_pool)
    closest_even_elem = num_pool * math.ceil(num_elementIDs / num_pool)
    det_diff = closest_even_det - num_detectors
    elem_diff = closest_even_elem - num_elementIDs
    padding = (
        (det_diff // 2, det_diff - det_diff // 2),
        (elem_diff // 2, elem_diff - elem_diff // 2),
    )

    x = layers.ZeroPadding2D(padding=padding)(input_layer)

    # Encoder
    if backbone == "resnet50":
        x = layers.Concatenate()([x, x, x])
        backbone = ResNet50(include_top=False, input_tensor=x, weights=None)

        # Partially freeze backbone and alter stride for compatibility with U-Net decoder
        # backbone.trainable = False
        for layer in backbone.layers:
            if layer.name == "conv1_conv":
                layer.strides = (1, 1)
            # if layer.name.startswith('conv5_') or layer.name.startswith('conv4_'):
            #     layer.trainable = True

        enc1 = backbone.get_layer("conv1_relu").output
        enc2 = backbone.get_layer("conv2_block3_out").output
        enc3 = backbone.get_layer("conv3_block4_out").output
        enc4 = backbone.get_layer("conv4_block6_out").output
        enc5 = backbone.get_layer("conv5_block3_out").output
    else:
        enc1 = conv_block(x, filters[0], use_bn=use_bn)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(enc1)

        enc2 = conv_block(pool1, filters[1], use_bn=use_bn)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(enc2)

        enc3 = conv_block(pool2, filters[2], use_bn=use_bn)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(enc3)

        enc4 = conv_block(pool3, filters[3], use_bn=use_bn, dropout_enc=dropout_enc)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(enc4)

        # Bottleneck
        enc5 = conv_block(pool4, filters[4], use_bn=use_bn, dropout_bn=dropout_bn)

    # Decoder
    dec1 = layers.Conv2DTranspose(filters[3], kernel_size=3, strides=2, padding="same")(
        enc5
    )  # padding same avoids cropping
    dec1 = layers.concatenate([dec1, enc4])  # skip connections
    dec1 = conv_block(dec1, filters[3], use_bn=use_bn)

    dec2 = layers.Conv2DTranspose(filters[2], kernel_size=3, strides=2, padding="same")(
        dec1
    )  # padding same avoids cropping
    dec2 = layers.concatenate([dec2, enc3])  # skip connections
    dec2 = conv_block(dec2, filters[2], use_bn=use_bn)

    dec3 = layers.Conv2DTranspose(filters[1], kernel_size=3, strides=2, padding="same")(
        dec2
    )  # padding same avoids cropping
    dec3 = layers.concatenate([dec3, enc2])  # skip connections
    dec3 = conv_block(dec3, filters[1], use_bn=use_bn)

    dec4 = layers.Conv2DTranspose(filters[0], kernel_size=3, strides=2, padding="same")(
        dec3
    )
    dec4 = layers.concatenate([dec4, enc1])  # skip connections
    dec4 = layers.Cropping2D(cropping=padding)(dec4)  # Remove extra padding
    dec4 = conv_block(dec4, filters[0], use_bn=use_bn)

    return dec4


def unetpp_backbone(
    input_layer: layers.Input,
    num_detectors: int,
    num_elementIDs: int,
    use_bn: bool,
    dropout_bn: float,
    dropout_enc: float,
    base: int,
    use_attn: bool = False,
    use_attn_ffn: bool = True,
    dropout_attn: float = 0.0,
) -> layers.Layer:
    """
    U-Net++ backbone.

    Args:
        input_layer (layers.Input): Input layer to the backbone.
        num_detectors (int): Number of detector channels in the input.
        num_elementIDs (int): Number of element ID channels in the input.
        use_bn (bool): Whether to use batch normalization.
        dropout_bn (float): Dropout rate after bottleneck if using batch normalization.
        dropout_enc (float): Dropout rate in the encoder.
        base (int): Base number of filters for the U-Net++.
        use_attn (bool): Whether to use axial attention after the decoder.
        use_attn_ffn (bool): Whether to use feed-forward network in axial attention.
        dropout_attn (float): Dropout rate in axial attention.

    Returns:
        layers.Layer: Output layer of the U-Net++ backbone.
    """

    # Zero padding (aligns to closest 2^n -> preserves input shape)
    filters = [base, base * 2, base * 4, base * 8, base * 16]
    num_pool = 2 ** (len(filters) - 1)  # 2 ^ n, n = number of max pooling
    closest_even_det = num_pool * math.ceil(num_detectors / num_pool)
    closest_even_elem = num_pool * math.ceil(num_elementIDs / num_pool)
    det_diff = closest_even_det - num_detectors
    elem_diff = closest_even_elem - num_elementIDs
    padding = (
        (det_diff // 2, det_diff - det_diff // 2),
        (elem_diff // 2, elem_diff - elem_diff // 2),
    )

    x = layers.ZeroPadding2D(padding=padding)(input_layer)

    # Encoder (starting with column j=0)
    X = [[None] * len(filters) for _ in range(len(filters))]  # X[i][j]

    X[0][0] = conv_block(x, filters[0], use_bn=use_bn)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(X[0][0])

    X[1][0] = conv_block(pool1, filters[1], use_bn=use_bn, dropout=dropout_enc / 2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(X[1][0])

    X[2][0] = conv_block(pool2, filters[2], use_bn=use_bn, dropout=dropout_enc)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(X[2][0])

    X[3][0] = conv_block(pool3, filters[3], use_bn=use_bn, dropout=dropout_enc)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(X[3][0])

    X[4][0] = conv_block(pool4, filters[4], use_bn=use_bn, dropout_bn=dropout_bn)

    # Decoder with dense skip connections
    for j in range(1, len(filters)):
        for i in range(0, len(filters) - j):
            concat_parts = [X[i][k] for k in range(j)] + [upsample(X[i + 1][j - 1])]
            X[i][j] = conv_block(
                layers.Concatenate()(concat_parts), filters[i], use_bn=use_bn
            )

    # Cropping to match input shape
    x = layers.Cropping2D(cropping=padding)(X[0][len(filters) - 1])

    # Optional axial attention along height and width axes
    if use_attn:
        x = AxialAttention(
            embed_dim=filters[0],
            axis="height",
            dropout=dropout_attn,
            use_ffn=use_attn_ffn,
        )(x)
        x = AxialAttention(
            embed_dim=filters[0],
            axis="width",
            dropout=dropout_attn,
            use_ffn=use_attn_ffn,
        )(x)

    return x
