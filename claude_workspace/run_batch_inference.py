"""
Batch inference script for AI Voice Detection.

Usage:
    python run_batch_inference.py <audio_directory>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.predictor import VoicePredictor


def main():
    audio_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    model_path = r"D:\datasets\models\best_model.keras"

    # Find all audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Loading model: {model_path}")
    predictor = VoicePredictor(model_path, threshold=0.5)

    print(f"\nProcessing {len(audio_files)} audio files...\n")
    print(f"{'File':<40} {'Result':<15} {'Probability':>12} {'Time (ms)':>10}")
    print("=" * 80)

    results = {"Human": 0, "AI-Generated": 0}

    for audio_file in sorted(audio_files):
        try:
            probability, label, inference_time = predictor.predict_file(str(audio_file))
            results[label] += 1
            print(f"{audio_file.name:<40} {label:<15} {probability:>11.2%} {inference_time:>10.2f}")
        except Exception as e:
            print(f"{audio_file.name:<40} ERROR: {e}")

    print("=" * 80)
    print(f"\nSummary: {results['Human']} Human, {results['AI-Generated']} AI-Generated")


if __name__ == "__main__":
    main()
