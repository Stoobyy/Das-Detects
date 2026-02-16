"""
TensorFlow Lite model conversion utilities.

Converts Keras models to TFLite format with optional quantization.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Generator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def create_representative_dataset(
    data_dir: Optional[str] = None,
    num_samples: int = config.REPRESENTATIVE_DATASET_SIZE,
) -> Callable[[], Generator]:
    """
    Create representative dataset generator for quantization.

    Args:
        data_dir: Path to data directory (uses random data if None)
        num_samples: Number of samples to generate

    Returns:
        Representative dataset generator function
    """
    if data_dir is not None:
        from data import DatasetLoader

        loader = DatasetLoader(data_dir, augment=False)
        features, _ = loader.load_all_data(max_samples_per_class=num_samples // 2)

        def representative_dataset():
            for i in range(min(num_samples, len(features))):
                yield [features[i:i+1].astype(np.float32)]

    else:
        # Use random data if no data directory provided
        input_shape = config.INPUT_SHAPE

        def representative_dataset():
            for _ in range(num_samples):
                data = np.random.randn(1, *input_shape).astype(np.float32)
                yield [data]

    return representative_dataset


def convert_to_tflite(
    model_path: str,
    output_path: Optional[str] = None,
    quantize: bool = True,
    data_dir: Optional[str] = None,
    optimization: str = "default",
) -> str:
    """
    Convert Keras model to TFLite format.

    Args:
        model_path: Path to Keras model (.keras)
        output_path: Output path for TFLite model
        quantize: Apply int8 quantization
        data_dir: Data directory for representative dataset
        optimization: Optimization mode ('default', 'size', 'latency')

    Returns:
        Path to saved TFLite model
    """
    import tensorflow as tf
    from tensorflow import keras
    from models.layers import DepthwiseSeparableConv2D, ConvBlock

    # Load Keras model
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
            "ConvBlock": ConvBlock,
        },
    )

    print(f"Loaded model from: {model_path}")
    print(f"Input shape: {model.input_shape}")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set optimizations
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Create representative dataset for full integer quantization
        representative_data = create_representative_dataset(data_dir)
        converter.representative_dataset = representative_data

        # Full integer quantization settings
        if optimization == "size":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            # Dynamic range quantization (better compatibility)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]

    # Convert
    print("Converting model to TFLite...")
    tflite_model = converter.convert()

    # Determine output path
    if output_path is None:
        model_path = Path(model_path)
        suffix = "_quantized" if quantize else ""
        output_path = model_path.parent / f"{model_path.stem}{suffix}.tflite"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    # Report sizes
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    tflite_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\nConversion complete!")
    print(f"  Original model: {original_size:.2f} MB")
    print(f"  TFLite model: {tflite_size:.2f} MB")
    print(f"  Compression: {(1 - tflite_size/original_size)*100:.1f}%")
    print(f"  Saved to: {output_path}")

    return str(output_path)


def verify_tflite_model(
    tflite_path: str,
    test_input: Optional[np.ndarray] = None,
) -> dict:
    """
    Verify TFLite model by running inference.

    Args:
        tflite_path: Path to TFLite model
        test_input: Test input array (uses random if None)

    Returns:
        Dict with verification results
    """
    import tensorflow as tf

    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test input
    if test_input is None:
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        test_input = np.random.randn(*input_shape).astype(input_dtype)

    # Run inference
    import time
    start = time.perf_counter()

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    inference_time = (time.perf_counter() - start) * 1000

    return {
        "input_shape": input_details[0]["shape"].tolist(),
        "input_dtype": str(input_details[0]["dtype"]),
        "output_shape": output_details[0]["shape"].tolist(),
        "output_dtype": str(output_details[0]["dtype"]),
        "output_value": float(output[0][0]),
        "inference_time_ms": inference_time,
    }


class TFLiteConverter:
    """
    TFLite conversion manager with validation and benchmarking.
    """

    def __init__(
        self,
        model_path: str,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize TFLiteConverter.

        Args:
            model_path: Path to Keras model
            data_dir: Path to data directory for calibration
        """
        self.model_path = Path(model_path)
        self.data_dir = data_dir
        self.tflite_path = None

    def convert(
        self,
        output_path: Optional[str] = None,
        quantize: bool = True,
    ) -> str:
        """
        Convert model to TFLite.

        Args:
            output_path: Output path for TFLite model
            quantize: Apply quantization

        Returns:
            Path to TFLite model
        """
        self.tflite_path = convert_to_tflite(
            str(self.model_path),
            output_path,
            quantize,
            self.data_dir,
        )
        return self.tflite_path

    def verify(self) -> dict:
        """
        Verify converted model.

        Returns:
            Verification results
        """
        if self.tflite_path is None:
            raise ValueError("No TFLite model to verify. Run convert() first.")

        return verify_tflite_model(self.tflite_path)

    def benchmark(self, n_iterations: int = 100) -> dict:
        """
        Benchmark TFLite inference speed.

        Args:
            n_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        import tensorflow as tf
        import time

        if self.tflite_path is None:
            raise ValueError("No TFLite model to benchmark. Run convert() first.")

        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Create test input
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]
        test_input = np.random.randn(*input_shape).astype(input_dtype)

        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            times.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "iterations": n_iterations,
        }

    def compare_with_keras(
        self,
        n_samples: int = 10,
    ) -> dict:
        """
        Compare TFLite output with original Keras model.

        Args:
            n_samples: Number of samples to compare

        Returns:
            Comparison results
        """
        import tensorflow as tf
        from tensorflow import keras
        from models.layers import DepthwiseSeparableConv2D, ConvBlock

        if self.tflite_path is None:
            raise ValueError("No TFLite model to compare. Run convert() first.")

        # Load Keras model
        keras_model = keras.models.load_model(
            str(self.model_path),
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Compare outputs
        input_shape = input_details[0]["shape"]
        differences = []

        for _ in range(n_samples):
            test_input = np.random.randn(*input_shape).astype(np.float32)

            # Keras prediction
            keras_output = keras_model.predict(test_input, verbose=0)[0][0]

            # TFLite prediction
            interpreter.set_tensor(input_details[0]["index"], test_input)
            interpreter.invoke()
            tflite_output = interpreter.get_tensor(output_details[0]["index"])[0][0]

            differences.append(abs(keras_output - tflite_output))

        return {
            "mean_difference": float(np.mean(differences)),
            "max_difference": float(np.max(differences)),
            "std_difference": float(np.std(differences)),
            "samples_compared": n_samples,
        }
