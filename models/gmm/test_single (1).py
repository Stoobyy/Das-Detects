"""
Single-file GMM+LFCC inference
Usage: python test_single.py
"""
import sys
import numpy as np
import librosa
from scipy.fft import dct
import joblib

# ── Config ──────────────────────────────────────────────
AUDIO_FILE  = "test human.wav"   # <-- change this as needed
MODEL_HUMAN = "gmm_human.pkl"
MODEL_AI    = "gmm_ai.pkl"

SAMPLE_RATE = 16_000
DURATION    = 2.5
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS      = 128
N_LFCC      = 20
N_FFT       = 2048
HOP_LENGTH  = 512
FMIN        = 80
FMAX        = 7500
TARGET_DB   = -20.0

# ── Helpers ──────────────────────────────────────────────
def load_audio(filepath):
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    return audio[:NUM_SAMPLES]


def rms_normalize(audio, target_db=TARGET_DB):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    return audio * (10 ** (target_db / 20.0) / rms)


def extract_lfcc(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    log_mel = np.log(mel + 1e-9)
    lfcc    = dct(log_mel, type=2, axis=0, norm="ortho")[:N_LFCC, :]
    return lfcc.T.astype(np.float64)   # (T, N_LFCC)


# ── Main ─────────────────────────────────────────────────
print("=" * 55)
print("  GMM + LFCC  |  Single File Inference")
print("=" * 55)
print(f"  File : {AUDIO_FILE}")

# Load models
try:
    gmm_human = joblib.load(MODEL_HUMAN)
    gmm_ai    = joblib.load(MODEL_AI)
    print(f"  Models loaded: {MODEL_HUMAN}, {MODEL_AI}")
except FileNotFoundError as e:
    sys.exit(f"ERROR: Model not found — {e}. Run gmm_lfcc_classifier.py first.")

# Load and process audio
try:
    audio  = load_audio(AUDIO_FILE)
    audio  = rms_normalize(audio)
    frames = extract_lfcc(audio)
    print(f"  Duration   : {DURATION:.1f}s  |  Frames : {frames.shape[0]}")
except Exception as e:
    sys.exit(f"ERROR: Could not load audio — {e}")

# Score
score_human = gmm_human.score(frames)
score_ai    = gmm_ai.score(frames)
llr         = score_ai - score_human

prediction  = "AI" if llr > 0 else "Human"
confidence  = "strong" if abs(llr) > 1.0 else "borderline"

# Display result
print()
print(f"  Score (Human GMM) : {score_human:>10.4f}")
print(f"  Score (AI GMM)    : {score_ai:>10.4f}")
print(f"  LLR  (AI - Human) : {llr:>+10.4f}")
print()
print(f"  >>> PREDICTION : {prediction} <<<")
print(f"  Confidence     : {abs(llr):.4f}  ({confidence})")
print("=" * 55)
