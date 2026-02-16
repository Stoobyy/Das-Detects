"""
Custom layers for the lightweight CNN architecture.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


class DepthwiseSeparableConv2D(layers.Layer):
    """
    Depthwise Separable Convolution layer.

    This layer performs a depthwise convolution followed by a pointwise
    convolution, reducing parameters by 8-10x compared to standard Conv2D.
    This is the key building block for MobileNet-style architectures.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: tuple = (3, 3),
        strides: tuple = (1, 1),
        padding: str = "same",
        use_batch_norm: bool = True,
        activation: str = "relu6",
        **kwargs
    ):
        """
        Initialize DepthwiseSeparableConv2D layer.

        Args:
            filters: Number of output filters
            kernel_size: Size of the depthwise kernel
            strides: Stride of the depthwise convolution
            padding: Padding mode ('same' or 'valid')
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu6' recommended for quantization)
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation

        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not use_batch_norm,
        )

        # Pointwise convolution (1x1 conv)
        self.pointwise_conv = layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=not use_batch_norm,
        )

        # Batch normalization layers
        if use_batch_norm:
            self.bn_depthwise = layers.BatchNormalization()
            self.bn_pointwise = layers.BatchNormalization()

        # Activation
        if activation == "relu6":
            self.activation = layers.ReLU(max_value=6.0)
        elif activation == "relu":
            self.activation = layers.ReLU()
        else:
            self.activation = layers.Activation(activation)

    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.depthwise_conv(inputs)
        if self.use_batch_norm:
            x = self.bn_depthwise(x, training=training)
        x = self.activation(x)

        x = self.pointwise_conv(x)
        if self.use_batch_norm:
            x = self.bn_pointwise(x, training=training)
        x = self.activation(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation_name,
        })
        return config


class ConvBlock(layers.Layer):
    """
    Standard convolutional block with BatchNorm and activation.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: tuple = (3, 3),
        strides: tuple = (1, 1),
        padding: str = "same",
        use_batch_norm: bool = True,
        activation: str = "relu6",
        **kwargs
    ):
        """
        Initialize ConvBlock.

        Args:
            filters: Number of output filters
            kernel_size: Size of the kernel
            strides: Stride of the convolution
            padding: Padding mode
            use_batch_norm: Whether to use batch normalization
            activation: Activation function
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation

        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not use_batch_norm,
        )

        if use_batch_norm:
            self.bn = layers.BatchNormalization()

        if activation == "relu6":
            self.activation = layers.ReLU(max_value=6.0)
        elif activation == "relu":
            self.activation = layers.ReLU()
        else:
            self.activation = layers.Activation(activation)

    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.conv(inputs)
        if self.use_batch_norm:
            x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation_name,
        })
        return config
