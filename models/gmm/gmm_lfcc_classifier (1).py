"""
=============================================================
 GMM + LFCC Baseline: Human vs AI Speech Classification
=============================================================
 Dependencies: numpy, librosa, scipy, scikit-learn, joblib
 Usage: python gmm_lfcc_classifier.py
=============================================================
"""

import os
import sys
import warnings
import numpy as np
import librosa
from scipy.fft import dct
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_HUMAN_DIR = "data/human"
TRAIN_AI_DIR    = "data/ai"
TEST_HUMAN_DIR  = "test/human"
TEST_AI_DIR     = "test/ai"

SAMPLE_RATE  = 16_000
DURATION     = 2.5                      # seconds
NUM_SAMPLES  = int(SAMPLE_RATE * DURATION)  # 40 000 samples
TARGET_DB    = -20.0                    # RMS normalisation target

# LFCC parameters
N_MELS      = 128
N_LFCC      = 20
N_FFT       = 2048
HOP_LENGTH  = 512
FMIN        = 80
FMAX        = 7500

# GMM parameters
N_COMPONENTS = 16
COV_TYPE     = "diag"
RANDOM_STATE = 42
MAX_ITER     = 200

MODEL_HUMAN = "gmm_human.pkl"
MODEL_AI    = "gmm_ai.pkl"

LABEL_HUMAN = 0
LABEL_AI    = 1
LABEL_NAMES = {LABEL_HUMAN: "Human", LABEL_AI: "AI"}

# ─────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────

def load_audio(filepath: str) -> np.ndarray:
    """Load mono audio at 16 kHz, pad or trim to exactly DURATION seconds."""
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    if len(audio) < NUM_SAMPLES:
        # Pad with zeros on the right
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    return audio


def rms_normalize(audio: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
    """Normalize audio to a target RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:          # silent audio – return as-is
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


# ─────────────────────────────────────────────
# LFCC FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_lfcc(audio: np.ndarray) -> np.ndarray:
    """
    Compute LFCC features.

    Pipeline:
        audio → linear filterbank mel-spectrogram → log power → DCT (type-II)

    Returns
    -------
    lfcc : np.ndarray, shape (T, N_LFCC)
        One feature vector per time frame.
    """
    # Step 1: Linear (mel) spectrogram  [shape: (N_MELS, T)]
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # Step 2: Log power  (add small epsilon for numerical stability)
    log_spec = np.log(mel_spec + 1e-9)   # shape: (N_MELS, T)

    # Step 3: DCT type-II along frequency axis → LFCC  [shape: (N_LFCC, T)]
    lfcc = dct(log_spec, type=2, axis=0, norm="ortho")[:N_LFCC, :]

    return lfcc.T.astype(np.float64)   # shape: (T, N_LFCC)


# ─────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────

SUPPORTED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

def list_audio_files(directory: str) -> list:
    """Return sorted list of audio file paths in a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )


def load_features_from_dir(directory: str, label_name: str) -> np.ndarray:
    """
    Load all audio files from *directory*, extract LFCC frames,
    and stack them into a single feature matrix.
    """
    files = list_audio_files(directory)
    if not files:
        print(f"  [WARNING] No audio files found in '{directory}'")
        return np.empty((0, N_LFCC))

    all_frames = []
    for i, fp in enumerate(files, 1):
        try:
            audio  = load_audio(fp)
            audio  = rms_normalize(audio)
            frames = extract_lfcc(audio)
            all_frames.append(frames)
            if i % 20 == 0 or i == len(files):
                print(f"  [{label_name}] Processed {i:>4d}/{len(files)} files …")
        except Exception as exc:
            print(f"  [SKIP] {os.path.basename(fp)}: {exc}")

    if not all_frames:
        return np.empty((0, N_LFCC))

    return np.vstack(all_frames)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_gmm(features: np.ndarray, label: str) -> GaussianMixture:
    """Fit a GMM on the given feature matrix."""
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COV_TYPE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    print(f"  Fitting GMM for [{label}] on {features.shape[0]:,} frames …")
    gmm.fit(features)
    print(f"  ✔  GMM [{label}] converged: {gmm.converged_}")
    return gmm


def run_training() -> tuple:
    """Full training pipeline. Returns (gmm_human, gmm_ai)."""
    print("\n" + "=" * 60)
    print("  PHASE 1 — TRAINING")
    print("=" * 60)

    # ── Human ──
    print(f"\n[1/4] Loading Human training data from '{TRAIN_HUMAN_DIR}' …")
    feats_human = load_features_from_dir(TRAIN_HUMAN_DIR, "Human")
    if feats_human.shape[0] == 0:
        sys.exit("ERROR: No human training samples found. Aborting.")

    # ── AI ──
    print(f"\n[2/4] Loading AI training data from '{TRAIN_AI_DIR}' …")
    feats_ai = load_features_from_dir(TRAIN_AI_DIR, "AI")
    if feats_ai.shape[0] == 0:
        sys.exit("ERROR: No AI training samples found. Aborting.")

    print(f"\n  Total frames  →  Human: {feats_human.shape[0]:,}  |  "
          f"AI: {feats_ai.shape[0]:,}")

    # ── Train GMMs ──
    print("\n[3/4] Training GMMs …")
    gmm_human = train_gmm(feats_human, "Human")
    gmm_ai    = train_gmm(feats_ai,    "AI")

    # ── Save models ──
    print("\n[4/4] Saving models …")
    joblib.dump(gmm_human, MODEL_HUMAN)
    joblib.dump(gmm_ai,    MODEL_AI)
    print(f"  Saved → {MODEL_HUMAN}")
    print(f"  Saved → {MODEL_AI}")

    return gmm_human, gmm_ai


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def predict_file(filepath: str, gmm_human: GaussianMixture,
                 gmm_ai: GaussianMixture) -> dict:
    """
    Predict a single audio file.

    Returns a dict with keys:
        filename, score_human, score_ai, prediction
    """
    audio  = load_audio(filepath)
    audio  = rms_normalize(audio)
    frames = extract_lfcc(audio)   # (T, N_LFCC)

    score_human = gmm_human.score(frames)   # mean log-likelihood
    score_ai    = gmm_ai.score(frames)

    prediction = LABEL_AI if score_ai > score_human else LABEL_HUMAN

    return {
        "filename":     os.path.basename(filepath),
        "score_human":  score_human,
        "score_ai":     score_ai,
        "prediction":   prediction,
    }


def evaluate_directory(directory: str, true_label: int,
                        gmm_human: GaussianMixture,
                        gmm_ai:    GaussianMixture) -> list:
    """
    Run prediction on every file in *directory*.
    Returns a list of result dicts (each includes 'true_label').
    """
    files   = list_audio_files(directory)
    results = []

    for i, fp in enumerate(files, 1):
        try:
            r = predict_file(fp, gmm_human, gmm_ai)
            r["true_label"] = true_label
            results.append(r)
            if i % 20 == 0 or i == len(files):
                print(f"  [{LABEL_NAMES[true_label]}] Evaluated "
                      f"{i:>4d}/{len(files)} files …")
        except Exception as exc:
            print(f"  [SKIP] {os.path.basename(fp)}: {exc}")

    return results


def print_evaluation_report(results: list) -> None:
    """Pretty-print the full evaluation report."""
    if not results:
        print("No evaluation results to report.")
        return

    y_true = [r["true_label"]  for r in results]
    y_pred = [r["prediction"]  for r in results]

    overall_acc = accuracy_score(y_true, y_pred)

    human_results = [r for r in results if r["true_label"] == LABEL_HUMAN]
    ai_results    = [r for r in results if r["true_label"] == LABEL_AI]

    human_acc = (
        sum(r["prediction"] == LABEL_HUMAN for r in human_results) / len(human_results)
        if human_results else float("nan")
    )
    ai_acc = (
        sum(r["prediction"] == LABEL_AI for r in ai_results) / len(ai_results)
        if ai_results else float("nan")
    )

    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_HUMAN, LABEL_AI])

    # ── Summary ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total files evaluated : {len(results)}")
    print(f"  Human test samples    : {len(human_results)}")
    print(f"  AI test samples       : {len(ai_results)}")
    print()
    print(f"  Overall Accuracy      : {overall_acc * 100:.2f} %")
    print(f"  Human Accuracy        : {human_acc  * 100:.2f} %")
    print(f"  AI Accuracy           : {ai_acc     * 100:.2f} %")

    # ── Confusion matrix ─────────────────────────
    print()
    print("  Confusion Matrix (rows=True, cols=Predicted):")
    print("                    Pred:Human   Pred:AI")
    print(f"  True:Human      {cm[0, 0]:>10d}  {cm[0, 1]:>9d}")
    print(f"  True:AI         {cm[1, 0]:>10d}  {cm[1, 1]:>9d}")

    # ── Per-file results ──────────────────────────
    COL = 40   # filename column width
    print()
    print("  Per-File Results:")
    print(f"  {'Filename':<{COL}} {'True':>6} {'Pred':>6} "
          f"{'Score_Human':>12} {'Score_AI':>10}")
    print("  " + "-" * (COL + 38))

    for r in results:
        match = "✔" if r["true_label"] == r["prediction"] else "✘"
        print(
            f"  {r['filename']:<{COL}} "
            f"{LABEL_NAMES[r['true_label']]:>6} "
            f"{LABEL_NAMES[r['prediction']]:>6} "
            f"{r['score_human']:>12.4f} "
            f"{r['score_ai']:>10.4f}  {match}"
        )

    print("=" * 60)


def run_evaluation(gmm_human: GaussianMixture, gmm_ai: GaussianMixture) -> None:
    """Full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("  PHASE 2 — EVALUATION")
    print("=" * 60)

    print(f"\n[1/2] Evaluating Human test data from '{TEST_HUMAN_DIR}' …")
    human_results = evaluate_directory(TEST_HUMAN_DIR, LABEL_HUMAN,
                                       gmm_human, gmm_ai)

    print(f"\n[2/2] Evaluating AI test data from '{TEST_AI_DIR}' …")
    ai_results    = evaluate_directory(TEST_AI_DIR, LABEL_AI,
                                       gmm_human, gmm_ai)

    all_results = human_results + ai_results

    if not all_results:
        print("\n[WARNING] No test files could be evaluated.")
        return

    print_evaluation_report(all_results)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  GMM + LFCC  |  Human vs AI Speech Classifier")
    print("=" * 60)
    print(f"  Training dirs : {TRAIN_HUMAN_DIR}  /  {TRAIN_AI_DIR}")
    print(f"  Test dirs     : {TEST_HUMAN_DIR}   /  {TEST_AI_DIR}")
    print(f"  Sample rate   : {SAMPLE_RATE} Hz")
    print(f"  Duration      : {DURATION} s  ({NUM_SAMPLES} samples)")
    print(f"  LFCC          : n_mels={N_MELS}, n_lfcc={N_LFCC}, "
          f"n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"  GMM           : components={N_COMPONENTS}, cov={COV_TYPE}")

    # ── Load models if they already exist, otherwise train ──
    if os.path.isfile(MODEL_HUMAN) and os.path.isfile(MODEL_AI):
        print(f"\n[INFO] Pre-trained models found. Loading …")
        gmm_human = joblib.load(MODEL_HUMAN)
        gmm_ai    = joblib.load(MODEL_AI)
        print(f"  Loaded → {MODEL_HUMAN}")
        print(f"  Loaded → {MODEL_AI}")
    else:
        print("\n[INFO] No pre-trained models found. Starting training …")
        gmm_human, gmm_ai = run_training()

    # ── Evaluate ──
    run_evaluation(gmm_human, gmm_ai)

    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
