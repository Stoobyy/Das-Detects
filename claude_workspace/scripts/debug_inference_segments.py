"""
Debug inference by analyzing audio in segments.
Determines if the model is triggering on speech or silence.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor
import config
from models.layers import DepthwiseSeparableConv2D, ConvBlock

def predict_segments(model, extractor, file_path):
    print(f"\n--- Analyzing Segments: {file_path.name} ---")

    # Load raw audio first to find speech/silence
    y, sr = librosa.load(str(file_path), sr=config.SAMPLE_RATE)
    duration = len(y) / sr
    print(f"Duration: {duration:.2f}s")

    # Split into 2.5s windows with 50% overlap
    window_samples = config.AUDIO_SAMPLES # 40000
    hop_samples = window_samples // 2

    windows = []
    timestamps = []

    for i in range(0, len(y) - window_samples + 1, hop_samples):
        windows.append(y[i:i+window_samples])
        timestamps.append(i / sr)

    if not windows:
        # File shorter than window? Pad it.
        windows.append(pad_or_trim(y, window_samples))
        timestamps.append(0.0)

    print(f"Analyzing {len(windows)} windows...")

    results = []
    for i, window in enumerate(windows):
        # 1. Measure energy (is this silence?)
        rms = np.sqrt(np.mean(window**2))
        is_silence = rms < 0.01
        status = "SILENCE" if is_silence else "SPEECH "

        # 2. Normalize (RMS)
        window_norm = normalize_audio(window, method="rms", target_db=-20.0)

        # 3. Extract features
        feats = extractor.extract(window_norm)
        feats = np.expand_dims(feats, axis=0)

        # 4. Predict
        prob = model.predict(feats, verbose=0)[0][0]
        label = "AI" if prob > 0.5 else "Human"

        results.append((timestamps[i], status, prob, label))

        # Visual bar
        bar = "#" * int(prob * 20)
        print(f"[{timestamps[i]:.1f}s] {status} | Prob: {prob:.2f} ({label}) | {bar}")

    return results

def main():
    test_file = Path(r"D:/datasets/LibriSpeech/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac")
    # test_file = Path(r"C:\Users\amrit\Downloads\testcases\annabel6_0007_211832_190.wav")
    model_path = r"D:\datasets\models_librispeech_opus_rms\best_model.keras"

    if not test_file.exists():
        print("Test file not found")
        return

    # Load Model
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    extractor = LFCCExtractor()

    predict_segments(model, extractor, test_file)

if __name__ == "__main__":
    main()
