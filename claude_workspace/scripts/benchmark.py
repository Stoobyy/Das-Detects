#!/usr/bin/env python3
"""
Benchmark script for inference speed testing.

Usage:
    python scripts/benchmark.py --model-path saved_models/model.tflite
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference import VoicePredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark AI voice detection inference"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file (.keras or .tflite)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )

    parser.add_argument(
        "--include-feature-extraction",
        action="store_true",
        help="Include feature extraction in timing",
    )

    return parser.parse_args()


def benchmark_full_pipeline(predictor: VoicePredictor, n_iterations: int) -> dict:
    """Benchmark full pipeline including feature extraction."""
    times = []
    dummy_audio = np.random.randn(config.AUDIO_SAMPLES).astype(np.float32)

    for _ in range(n_iterations):
        start = time.perf_counter()
        _, _, _ = predictor.predict_audio(dummy_audio)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def benchmark_feature_extraction(n_iterations: int) -> dict:
    """Benchmark feature extraction only."""
    from features import LFCCExtractor

    extractor = LFCCExtractor()
    dummy_audio = np.random.randn(config.AUDIO_SAMPLES).astype(np.float32)

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = extractor.extract(dummy_audio)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    """Main benchmark function."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("AI VOICE DETECTION - BENCHMARK")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Audio window: {config.AUDIO_DURATION}s @ {config.SAMPLE_RATE}Hz")
    print("=" * 60 + "\n")

    # Get model info
    model_path = Path(args.model_path)
    model_size = model_path.stat().st_size / (1024 * 1024)
    model_type = "TFLite" if model_path.suffix == ".tflite" else "Keras"

    print(f"Model type: {model_type}")
    print(f"Model size: {model_size:.2f} MB")

    # Initialize predictor
    print("\nLoading model...")
    predictor = VoicePredictor(str(model_path))

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        dummy = np.random.randn(config.AUDIO_SAMPLES).astype(np.float32)
        predictor.predict_audio(dummy)

    # Benchmark full pipeline
    print(f"\nBenchmarking full pipeline ({args.iterations} iterations)...")
    full_results = benchmark_full_pipeline(predictor, args.iterations)

    print("\n" + "-" * 40)
    print("FULL PIPELINE (audio -> prediction)")
    print("-" * 40)
    print(f"Mean:   {full_results['mean_ms']:7.2f} ms")
    print(f"Std:    {full_results['std_ms']:7.2f} ms")
    print(f"Min:    {full_results['min_ms']:7.2f} ms")
    print(f"Max:    {full_results['max_ms']:7.2f} ms")
    print(f"P50:    {full_results['p50_ms']:7.2f} ms")
    print(f"P95:    {full_results['p95_ms']:7.2f} ms")
    print(f"P99:    {full_results['p99_ms']:7.2f} ms")

    # Benchmark feature extraction separately
    if args.include_feature_extraction:
        print(f"\nBenchmarking feature extraction ({args.iterations} iterations)...")
        feat_results = benchmark_feature_extraction(args.iterations)

        print("\n" + "-" * 40)
        print("FEATURE EXTRACTION ONLY")
        print("-" * 40)
        print(f"Mean:   {feat_results['mean_ms']:7.2f} ms")
        print(f"Std:    {feat_results['std_ms']:7.2f} ms")

        # Estimate model-only time
        model_only = full_results['mean_ms'] - feat_results['mean_ms']
        print(f"\nEstimated model inference only: {model_only:.2f} ms")

    # Throughput
    throughput = 1000 / full_results['mean_ms']
    print(f"\nThroughput: {throughput:.1f} inferences/second")

    # Real-time factor
    rtf = full_results['mean_ms'] / (config.AUDIO_DURATION * 1000)
    print(f"Real-time factor: {rtf:.4f} (< 1.0 = real-time capable)")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)

    latency_ok = full_results['mean_ms'] < 100
    size_ok = model_size < 1.0
    realtime_ok = rtf < 1.0

    print(f"[{'OK' if latency_ok else 'FAIL'}] Inference < 100ms: {full_results['mean_ms']:.2f} ms")
    print(f"[{'OK' if size_ok else 'WARN'}] Model size < 1MB: {model_size:.2f} MB")
    print(f"[{'OK' if realtime_ok else 'FAIL'}] Real-time capable: RTF = {rtf:.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
