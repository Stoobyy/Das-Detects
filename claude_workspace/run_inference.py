"""
Simple inference script for AI Voice Detection.

Usage:
    python run_inference.py <audio_file>
    python run_inference.py <audio_file> --tflite
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.predictor import VoicePredictor


def main():
    parser = argparse.ArgumentParser(description="AI Voice Detection Inference")
    parser.add_argument("audio_file", help="Path to audio file to analyze")
    parser.add_argument(
        "--model",
        default=r"D:\datasets\models\best_model.keras",
        help="Path to model file",
    )
    parser.add_argument(
        "--tflite",
        action="store_true",
        help="Use TFLite model instead",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)",
    )
    args = parser.parse_args()

    # Use TFLite model if requested
    model_path = args.model
    if args.tflite:
        model_path = r"D:\datasets\models\model.tflite"

    print(f"Loading model: {model_path}")
    predictor = VoicePredictor(model_path, threshold=args.threshold)

    print(f"\nAnalyzing: {args.audio_file}")
    probability, label, inference_time = predictor.predict_file(args.audio_file)

    print(f"\n{'='*50}")
    print(f"Result: {label}")
    print(f"AI Probability: {probability:.2%}")
    print(f"Inference Time: {inference_time:.2f}ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
