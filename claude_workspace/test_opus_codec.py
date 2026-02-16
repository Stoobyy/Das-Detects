"""
Test: Apply Opus codec to training samples and compare to WhatsApp audio.

This tests the hypothesis that WhatsApp's Opus encoding creates artifacts
the model misclassifies as AI-generated.
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import wave
import numpy as np
from scipy import signal

sys.path.insert(0, '.')

from inference.predictor import VoicePredictor
import config


def load_wav(file_path: str, target_sr: int = 16000):
    """Load wav file and resample."""
    with wave.open(file_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

        # Convert to float32
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Stereo to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        # Resample
        if sr != target_sr:
            audio = signal.resample(audio, int(len(audio) * target_sr / sr))

        return audio.astype(np.float32), target_sr


def encode_opus(audio: np.ndarray, sample_rate: int = 16000, bitrate: str = "24k"):
    """Encode audio through Opus codec using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as wav
        input_path = Path(tmpdir) / "input.wav"
        opus_path = Path(tmpdir) / "encoded.opus"
        output_path = Path(tmpdir) / "output.wav"

        # Write input wav
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(str(input_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Encode to Opus
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:a", "libopus", "-b:a", bitrate,
            "-ar", str(sample_rate),
            str(opus_path)
        ], capture_output=True)

        # Decode back to wav
        subprocess.run([
            "ffmpeg", "-y", "-i", str(opus_path),
            "-ar", str(sample_rate),
            str(output_path)
        ], capture_output=True)

        # Load result
        encoded_audio, _ = load_wav(str(output_path), sample_rate)

        return encoded_audio


def main():
    # Load model
    print("Loading model...")
    predictor = VoicePredictor(r"D:\datasets\models\best_model.keras")

    # Test with a bonafide (human) sample from ASVspoof
    human_sample = r"D:\datasets\ASVspoof2021_LA_eval\flac\LA_E_5410749.flac"

    print(f"\nTesting: {human_sample}")

    # Load original audio
    import librosa
    audio_orig, _ = librosa.load(human_sample, sr=16000)

    # Test original
    prob_orig, label_orig, _ = predictor.predict_audio(audio_orig)
    print(f"\nOriginal (no codec):     {label_orig} ({prob_orig:.2%})")

    # Test with different Opus bitrates (WhatsApp uses ~24-32kbps)
    for bitrate in ["16k", "24k", "32k", "48k"]:
        audio_opus = encode_opus(audio_orig, 16000, bitrate)
        prob, label, _ = predictor.predict_audio(audio_opus)
        print(f"Opus @ {bitrate}:             {label} ({prob:.2%})")

    # Also test a spoof (AI) sample
    spoof_sample = r"D:\datasets\ASVspoof2021_LA_eval\flac\LA_E_6183634.flac"
    audio_spoof, _ = librosa.load(spoof_sample, sr=16000)

    print(f"\n\nSpoof sample: {spoof_sample}")
    prob_orig, label_orig, _ = predictor.predict_audio(audio_spoof)
    print(f"Original (no codec):     {label_orig} ({prob_orig:.2%})")

    for bitrate in ["16k", "24k", "32k"]:
        audio_opus = encode_opus(audio_spoof, 16000, bitrate)
        prob, label, _ = predictor.predict_audio(audio_opus)
        print(f"Opus @ {bitrate}:             {label} ({prob:.2%})")


if __name__ == "__main__":
    main()
