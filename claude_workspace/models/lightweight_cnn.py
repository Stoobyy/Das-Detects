"""
Lightweight CNN architecture for AI voice detection.

This MobileNet-style architecture uses depthwise separable convolutions
for efficient inference on CPU/mobile devices.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from typing import Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from .layers import DepthwiseSeparableConv2D, ConvBlock


def build_lightweight_cnn(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    filters: list = None,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units: int = config.DENSE_UNITS,
    label_smoothing: float = config.LABEL_SMOOTHING,
) -> Model:
    """
    Build a lightweight MobileNet-style CNN for voice detection.

    Architecture:
    - Initial Conv2D with stride 2 for downsampling
    - 3 DepthwiseSeparable blocks with MaxPooling
    - GlobalAveragePooling (no flatten needed)
    - Dense layer with dropout
    - Sigmoid output for binary classification

    Args:
        input_shape: Input tensor shape (n_lfcc, time_frames, channels)
        filters: List of filter counts for each block
        dropout_rate: Dropout rate before final dense layer
        dense_units: Number of units in dense layer
        label_smoothing: Label smoothing factor (applied during training)

    Returns:
        Compiled Keras model
    """
    if filters is None:
        filters = config.MODEL_FILTERS  # [16, 32, 64, 128]

    # Input layer
    inputs = layers.Input(shape=input_shape, name="input")

    # Block 1: Initial Conv2D with stride 2
    x = ConvBlock(
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        name="conv_block_1"
    )(inputs)

    # Block 2: DepthwiseSeparable + MaxPool
    x = DepthwiseSeparableConv2D(
        filters=filters[1],
        kernel_size=(3, 3),
        name="dw_sep_block_2"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)

    # Block 3: DepthwiseSeparable + MaxPool
    x = DepthwiseSeparableConv2D(
        filters=filters[2],
        kernel_size=(3, 3),
        name="dw_sep_block_3"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)

    # Block 4: DepthwiseSeparable + MaxPool
    x = DepthwiseSeparableConv2D(
        filters=filters[3],
        kernel_size=(3, 3),
        name="dw_sep_block_4"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_4")(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense layer with dropout
    x = layers.Dense(dense_units, name="dense")(x)
    x = layers.ReLU(max_value=6.0, name="dense_relu6")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs, name="lightweight_cnn")

    return model


def compile_model(
    model: Model,
    initial_lr: float = config.INITIAL_LEARNING_RATE,
    label_smoothing: float = config.LABEL_SMOOTHING,
) -> Model:
    """
    Compile model with optimizer and loss function.

    Args:
        model: Keras model to compile
        initial_lr: Initial learning rate
        label_smoothing: Label smoothing factor

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

    loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


class LightweightCNN:
    """
    Wrapper class for the lightweight CNN model with training utilities.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
        filters: list = None,
        dropout_rate: float = config.DROPOUT_RATE,
        dense_units: int = config.DENSE_UNITS,
    ):
        """
        Initialize LightweightCNN.

        Args:
            input_shape: Input tensor shape
            filters: List of filter counts for each block
            dropout_rate: Dropout rate
            dense_units: Number of units in dense layer
        """
        self.input_shape = input_shape
        self.filters = filters or config.MODEL_FILTERS
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

        self.model = None
        self._build_model()

    def _build_model(self):
        """Build and compile the model."""
        self.model = build_lightweight_cnn(
            input_shape=self.input_shape,
            filters=self.filters,
            dropout_rate=self.dropout_rate,
            dense_units=self.dense_units,
        )
        compile_model(self.model)

    def summary(self):
        """Print model summary."""
        self.model.summary()

    def get_model(self) -> Model:
        """Get the Keras model."""
        return self.model

    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "LightweightCNN":
        """
        Load model from file.

        Args:
            path: Path to saved model

        Returns:
            LightweightCNN instance with loaded model
        """
        instance = cls.__new__(cls)
        instance.model = keras.models.load_model(
            path,
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )
        return instance

    def count_parameters(self) -> dict:
        """
        Count model parameters.

        Returns:
            Dict with total, trainable, and non-trainable counts
        """
        trainable = sum(
            tf.keras.backend.count_params(w) for w in self.model.trainable_weights
        )
        non_trainable = sum(
            tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights
        )
        return {
            "total": trainable + non_trainable,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def estimate_size_mb(self) -> float:
        """
        Estimate model size in megabytes.

        Returns:
            Estimated size in MB
        """
        params = self.count_parameters()
        # Assuming float32 (4 bytes per parameter)
        return params["total"] * 4 / (1024 * 1024)


def create_model_with_summary() -> Model:
    """
    Create model and print summary for verification.

    Returns:
        Compiled Keras model
    """
    model = build_lightweight_cnn()
    compile_model(model)

    print("\n" + "=" * 60)
    print("LIGHTWEIGHT CNN FOR AI VOICE DETECTION")
    print("=" * 60)
    model.summary()

    # Count parameters
    wrapper = LightweightCNN()
    params = wrapper.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    print(f"\nEstimated model size: {wrapper.estimate_size_mb():.2f} MB")
    print("=" * 60 + "\n")

    return model


if __name__ == "__main__":
    # Test model creation
    create_model_with_summary()
