"""
Run inference using the RMS-normalized model (Opus 0.8).
"""

import sys
from pathlib import Path
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor
import config
from models.layers import DepthwiseSeparableConv2D, ConvBlock

def predict_file(model, extractor, file_path):
    try:
        # Load audio (automatically uses RMS norm from config.py)
        audio, _ = load_audio(str(file_path), config.SAMPLE_RATE)
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)

        # Explicitly verify normalization in debug
        # audio = normalize_audio(audio) # Already done in load_audio if updated? No, usually separate step.
        # Let's check features/audio_utils.py: load_audio just loads. We need to normalize.
        audio = normalize_audio(audio)

        features = extractor.extract(audio)
        features = np.expand_dims(features, axis=0)

        prob = model.predict(features, verbose=0)[0][0]
        label = "AI" if prob > 0.5 else "Human"
        return prob, label
    except Exception as e:
        return None, f"Error: {e}"

def main():
    test_dir = Path(r"C:\Users\amrit\Downloads\abraham")
    # New model path (Custom Data + LibriSpeech + RMS Norm)
    model_path = r"D:\datasets\models_custom_rms\best_model.keras"

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    test_files = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(test_files)} test files")
    print(f"Loading model: {model_path}")

    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    extractor = LFCCExtractor()

    print("\n" + "=" * 60)
    print("OPUS 0.8 (RMS Normalized) RESULTS")
    print("=" * 60)

    ai_count = 0
    human_count = 0

    for test_file in test_files:
        prob, label = predict_file(model, extractor, test_file)
        if prob is not None:
            # Color code output if supported, else just text
            print(f"{test_file.name:<45} | {label:>6} ({prob:.2f})")
            if label == "AI":
                ai_count += 1
            else:
                human_count += 1
        else:
            print(f"{test_file.name:<45} | Error")

    print("=" * 60)
    print(f"SUMMARY: {ai_count} AI, {human_count} Human")
    print(f"Normalization Method: {config.NORMALIZATION_METHOD} ({config.NORMALIZATION_TARGET_DB} dB)")

if __name__ == "__main__":
    main()
