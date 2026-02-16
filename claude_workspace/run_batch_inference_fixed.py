"""
Fixed batch inference script - handles 48kHz wav files properly.
"""

import sys
from pathlib import Path
import wave
import struct
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))

from inference.predictor import VoicePredictor
import config


def load_wav_scipy(file_path: str, target_sr: int = 16000):
    """Load wav file using wave module and resample."""
    with wave.open(file_path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        frames = wf.readframes(n_frames)

        # Convert bytes to numpy array
        if sample_width == 2:  # 16-bit
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:  # 32-bit
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)

        return audio.astype(np.float32), target_sr


def check_has_audio(file_path: str, threshold: float = 0.001) -> bool:
    """Check if file has actual audio content."""
    with wave.open(file_path, 'rb') as wf:
        n_frames = min(wf.getnframes(), 5000)
        frames = wf.readframes(n_frames)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return np.std(samples) > threshold


def main():
    audio_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    model_path = r"D:\datasets\models\best_model.keras"

    # Find all audio files
    audio_files = list(audio_dir.glob("*.wav"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Loading model: {model_path}")
    predictor = VoicePredictor(model_path, threshold=0.5)

    print(f"\nProcessing {len(audio_files)} audio files...\n")
    print(f"{'File':<40} {'Has Audio':<12} {'Result':<15} {'Probability':>12} {'Time (ms)':>10}")
    print("=" * 95)

    results = {"Human": 0, "AI-Generated": 0, "Empty": 0}

    for audio_file in sorted(audio_files):
        try:
            # Check if file has audio
            has_audio = check_has_audio(str(audio_file))

            if not has_audio:
                results["Empty"] += 1
                print(f"{audio_file.name:<40} {'No':<12} {'EMPTY FILE':<15} {'-':>12} {'-':>10}")
                continue

            # Load with scipy
            audio, sr = load_wav_scipy(str(audio_file), config.SAMPLE_RATE)

            # Run inference
            probability, label, inference_time = predictor.predict_audio(audio)
            results[label] += 1
            print(f"{audio_file.name:<40} {'Yes':<12} {label:<15} {probability:>11.2%} {inference_time:>10.2f}")

        except Exception as e:
            print(f"{audio_file.name:<40} {'?':<12} ERROR: {e}")

    print("=" * 95)
    print(f"\nSummary: {results['Human']} Human, {results['AI-Generated']} AI-Generated, {results['Empty']} Empty")


if __name__ == "__main__":
    main()
