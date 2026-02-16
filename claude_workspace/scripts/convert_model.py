#!/usr/bin/env python3
"""
TFLite conversion script.

Usage:
    python scripts/convert_model.py --model-path saved_models/best_model.keras --quantize
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference import TFLiteConverter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Keras model to TFLite"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Keras model (.keras file)",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for TFLite model",
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply int8 quantization",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for calibration (optional)",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted model",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with original Keras model",
    )

    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("TFLITE CONVERSION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Quantize: {args.quantize}")
    print("=" * 60 + "\n")

    # Initialize converter
    converter = TFLiteConverter(args.model_path, args.data_dir)

    # Convert
    tflite_path = converter.convert(
        output_path=args.output_path,
        quantize=args.quantize,
    )

    # Verify
    if args.verify:
        print("\n" + "-" * 40)
        print("VERIFICATION")
        print("-" * 40)
        results = converter.verify()
        print(f"Input shape: {results['input_shape']}")
        print(f"Input dtype: {results['input_dtype']}")
        print(f"Output shape: {results['output_shape']}")
        print(f"Output dtype: {results['output_dtype']}")
        print(f"Test output: {results['output_value']:.4f}")
        print(f"Inference time: {results['inference_time_ms']:.2f} ms")

    # Benchmark
    if args.benchmark:
        print("\n" + "-" * 40)
        print("BENCHMARK")
        print("-" * 40)
        results = converter.benchmark(n_iterations=100)
        print(f"Mean: {results['mean_ms']:.2f} ms")
        print(f"Std: {results['std_ms']:.2f} ms")
        print(f"Min: {results['min_ms']:.2f} ms")
        print(f"Max: {results['max_ms']:.2f} ms")
        print(f"P95: {results['p95_ms']:.2f} ms")
        print(f"P99: {results['p99_ms']:.2f} ms")

        # Check success criteria
        meets_latency = results['mean_ms'] < 100
        print(f"\n[{'OK' if meets_latency else 'FAIL'}] Latency < 100ms: {results['mean_ms']:.2f} ms")

    # Compare with Keras
    if args.compare:
        print("\n" + "-" * 40)
        print("KERAS COMPARISON")
        print("-" * 40)
        results = converter.compare_with_keras()
        print(f"Mean difference: {results['mean_difference']:.6f}")
        print(f"Max difference: {results['max_difference']:.6f}")
        print(f"Std difference: {results['std_difference']:.6f}")

    # Check file size
    tflite_size = Path(tflite_path).stat().st_size / (1024 * 1024)
    print(f"\n[{'OK' if tflite_size < 1.0 else 'WARN'}] Model size < 1MB: {tflite_size:.2f} MB")

    print("\n" + "=" * 60)
    print(f"TFLite model saved to: {tflite_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
