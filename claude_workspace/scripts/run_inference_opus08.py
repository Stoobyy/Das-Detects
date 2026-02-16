"""
Simple inference script for tf-gpu environment (TF 2.10).
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
        audio, _ = load_audio(str(file_path), config.SAMPLE_RATE)
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)
        audio = normalize_audio(audio)
        features = extractor.extract(audio)
        features = np.expand_dims(features, axis=0)
        prob = model.predict(features, verbose=0)[0][0]
        label = "AI" if prob > 0.5 else "Human"
        return prob, label
    except Exception as e:
        return None, f"Error: {e}"

def main():
    test_dir = Path(r"C:\Users\amrit\Downloads\testcases")
    model_path = r"D:\datasets\models_opus_0.8\best_model.keras"

    test_files = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(test_files)} test files")
    print(f"Loading model: {model_path}")

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
            "ConvBlock": ConvBlock,
        },
    )
    print("Model loaded!")

    extractor = LFCCExtractor()

    print("\n" + "=" * 60)
    print("OPUS 0.8 MODEL RESULTS")
    print("=" * 60)

    ai_count = 0
    human_count = 0

    for test_file in test_files:
        prob, label = predict_file(model, extractor, test_file)
        if prob is not None:
            print(f"{test_file.name:<45} | {label:>6} ({prob:.2f})")
            if label == "AI":
                ai_count += 1
            else:
                human_count += 1

    print("=" * 60)
    print(f"SUMMARY: {ai_count} AI, {human_count} Human")

if __name__ == "__main__":
    main()
