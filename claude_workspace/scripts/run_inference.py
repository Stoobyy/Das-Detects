"""
Run inference on test files using multiple models.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor
import config

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model(model_path):
    """Load a Keras model (handles both old HDF5 and new zip formats)."""
    from models.layers import DepthwiseSeparableConv2D, ConvBlock
    import h5py

    custom_objects = {
        "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
        "ConvBlock": ConvBlock,
    }

    # Check if it's HDF5 format (TF 2.10) or zip format (newer TF)
    try:
        with h5py.File(model_path, 'r') as f:
            # It's HDF5, use legacy loading
            return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except:
        # Try standard loading
        return keras.models.load_model(model_path, custom_objects=custom_objects)

def predict_file(model, extractor, file_path):
    """Run inference on a single file."""
    try:
        audio, _ = load_audio(str(file_path), config.SAMPLE_RATE)
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)
        audio = normalize_audio(audio)
        features = extractor.extract(audio)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        prob = model.predict(features, verbose=0)[0][0]
        label = "AI" if prob > 0.5 else "Human"
        return prob, label
    except Exception as e:
        return None, f"Error: {e}"

def main():
    test_dir = Path(r"C:\Users\amrit\Downloads\testcases")

    models = {
        "baseline (no opus)": r"D:\datasets\models_backup_v1\best_model.keras",
        "opus 0.5": r"D:\datasets\models\best_model.keras",
        "opus 0.8": r"D:\datasets\models_opus_0.8\best_model.keras",
    }

    # Get test files
    test_files = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(test_files)} test files\n")

    # Initialize feature extractor
    extractor = LFCCExtractor()

    # Load all models
    loaded_models = {}
    for name, path in models.items():
        print(f"Loading {name}...")
        try:
            loaded_models[name] = load_model(path)
        except Exception as e:
            print(f"  Failed to load: {e}")

    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)

    # Print header
    header = f"{'File':<45}"
    for name in loaded_models.keys():
        header += f" | {name:<18}"
    print(header)
    print("-" * 80)

    # Run inference
    for test_file in test_files:
        row = f"{test_file.name:<45}"
        for name, model in loaded_models.items():
            prob, label = predict_file(model, extractor, test_file)
            if prob is not None:
                row += f" | {label:>6} ({prob:.2f})    "
            else:
                row += f" | {'Error':<18}"
        print(row)

    print("=" * 80)

    # Summary per model
    print("\nSUMMARY:")
    for name, model in loaded_models.items():
        ai_count = 0
        human_count = 0
        for test_file in test_files:
            prob, label = predict_file(model, extractor, test_file)
            if label == "AI":
                ai_count += 1
            elif label == "Human":
                human_count += 1
        print(f"  {name}: {ai_count} AI, {human_count} Human")

if __name__ == "__main__":
    main()
