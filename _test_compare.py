"""Quick diagnostic: compare Keras vs TFLite on captured VoIP frames."""
import sys, os, glob, numpy as np
sys.path.insert(0, r"C:\Users\amrit\OneDrive\Documents\GitHub\Das-Detects\claude_workspace")

from models.layers import DepthwiseSeparableConv2D, ConvBlock
from features.mel_spectrogram import MelSpectrogramExtractor
from features.audio_utils import load_audio, pad_or_trim, normalize_audio
import config
import tensorflow as tf
import librosa

# Our TFLite module
sys.path.insert(0, r"C:\Users\amrit\OneDrive\Documents\GitHub\Das-Detects")
from modules.tflite_inferencer import TFLiteVoiceClassifier, silence_ratio

# Load Keras model
model = tf.keras.models.load_model(
    r"D:\datasets\models_mel_custom_v4\best_model.keras",
    custom_objects={"DepthwiseSeparableConv2D": DepthwiseSeparableConv2D, "ConvBlock": ConvBlock},
    compile=False,
)
extractor = MelSpectrogramExtractor()

# Load TFLite model
tflite = TFLiteVoiceClassifier(r"models\voice_classifier.tflite")

# Process all temp frames
files = sorted(glob.glob(r"temp\*.wav"))
print(f"Found {len(files)} frames\n")
print(f"{'FILE':<35s}  {'KERAS':>7s}  {'TFLITE':>7s}  {'SIL%':>5s}  {'DUR':>5s}")
print("-" * 70)

for f in files:
    # Raw audio at original SR
    raw, raw_sr = librosa.load(f, sr=None, mono=True)
    sil = silence_ratio(raw, sr=raw_sr)
    dur = len(raw) / raw_sr

    # Keras: use original training pipeline
    a16, _ = load_audio(f, config.SAMPLE_RATE)
    a16 = pad_or_trim(a16, config.AUDIO_SAMPLES)
    a16 = normalize_audio(a16)
    feat = np.expand_dims(extractor.extract(a16), 0)
    kp = float(model.predict(feat, verbose=0)[0][0]) * 100

    # TFLite: our module
    tc, tl, _ = tflite.classify(raw, source_sr=raw_sr)

    k_label = "AI" if kp >= 50 else "HUM"
    t_label = "AI" if tc >= 50 else "HUM"
    print(f"{os.path.basename(f):<35s}  {k_label} {kp:5.1f}%  {t_label} {tc:5.1f}%  {sil*100:4.0f}%  {dur:.1f}s")
