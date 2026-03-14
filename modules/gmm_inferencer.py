"""
GMM Voice Classifier — Real-time inference engine.

Loads pre-trained GMM models (Human + AI) and runs LFCC-based
AI vs Human voice classification on raw audio numpy arrays.

Feature extraction uses LFCC (Linear Frequency Cepstral Coefficients):
  SR=16 kHz, 2.5 s windows, N_MELS=128, N_LFCC=20
  → Scoring: log-likelihood comparison between human GMM & AI GMM
  → Output: probability mapped to 0-100 scale
"""

import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required: pip install librosa")

try:
    from scipy.fft import dct
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

try:
    import joblib
except ImportError:
    raise ImportError("joblib is required: pip install joblib")


# ── Feature-extraction constants (must match GMM training config) ────────
SAMPLE_RATE    = 16_000
AUDIO_DURATION = 2.5                                   # seconds
AUDIO_SAMPLES  = int(SAMPLE_RATE * AUDIO_DURATION)     # 40 000
TARGET_DB      = -20.0                                 # RMS normalisation

# LFCC parameters
N_MELS     = 128
N_LFCC     = 20
N_FFT      = 2048
HOP_LENGTH = 512
FMIN       = 80
FMAX       = 7500


# ── Audio helpers ────────────────────────────────────────────────────────

def _pad_or_trim(audio: np.ndarray, target_length: int = AUDIO_SAMPLES) -> np.ndarray:
    """Pad (right) or trim audio to exactly target_length samples."""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio[:target_length]


def _rms_normalize(audio: np.ndarray) -> np.ndarray:
    """RMS-normalise audio to TARGET_DB dBFS, matching training pipeline."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (TARGET_DB / 20.0)
    return audio * (target_rms / rms)


def _extract_lfcc(audio: np.ndarray) -> np.ndarray:
    """
    Compute LFCC features.

    Pipeline:
        audio → mel spectrogram → log power → DCT (type-II)

    Returns
    -------
    lfcc : np.ndarray, shape (T, N_LFCC)
        One feature vector per time frame.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    log_spec = np.log(mel_spec + 1e-9)                          # (N_MELS, T)
    lfcc = dct(log_spec, type=2, axis=0, norm="ortho")[:N_LFCC, :]  # (N_LFCC, T)
    return lfcc.T.astype(np.float64)                             # (T, N_LFCC)


# ── GMM inference wrapper ────────────────────────────────────────────────

class GMMVoiceClassifier:
    """
    GMM-based AI voice classifier using LFCC features.

    Uses two separately-trained GMMs (one for human speech, one for AI speech).
    Classification is based on which GMM assigns higher log-likelihood.

    Call `classify(audio, source_sr)` for synchronous inference,
    or use `submit()` + `get_result()` for background-thread inference.
    """

    def __init__(self, human_pkl: str, ai_pkl: str):
        self._gmm_human = joblib.load(human_pkl)
        self._gmm_ai = joblib.load(ai_pkl)
        print(f"[GMMVoiceClassifier] Loaded: {human_pkl}, {ai_pkl}")

        # Background inference thread
        self._task_queue: queue.Queue = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Warmup
        dummy = np.random.randn(AUDIO_SAMPLES).astype(np.float32)
        self.classify(dummy, source_sr=SAMPLE_RATE)
        print(f"[GMMVoiceClassifier] Warmed up and ready")

    # ── Synchronous inference ────────────────────────────────────────
    def classify(
        self,
        audio: np.ndarray,
        source_sr: int = 48_000,
    ) -> Tuple[float, str, float]:
        """
        Classify an audio array.

        Args:
            audio:     1-D float numpy array (mono).
            source_sr: Sample rate of the input audio.

        Returns:
            (confidence_pct, label, inference_ms)
            confidence_pct: 0-100 scale (higher = more likely AI).
            label:          "HUMAN" / "SUSPICIOUS" / "AI"
            inference_ms:   wall-clock time for inference.
        """
        # Resample to 16 kHz if needed
        if source_sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=source_sr, target_sr=SAMPLE_RATE)

        audio = _pad_or_trim(audio)
        audio = _rms_normalize(audio)
        lfcc = _extract_lfcc(audio)  # (T, N_LFCC)

        t0 = time.perf_counter()
        score_human = self._gmm_human.score(lfcc)  # mean log-likelihood
        score_ai = self._gmm_ai.score(lfcc)
        inference_ms = (time.perf_counter() - t0) * 1000

        # Convert log-likelihood difference to a 0-100 confidence scale.
        # diff > 0 → AI more likely;  diff < 0 → Human more likely
        diff = score_ai - score_human

        # Sigmoid mapping: maps unbounded diff to (0, 1)
        # Scale factor controls sensitivity (tuned for typical GMM score ranges)
        prob_ai = 1.0 / (1.0 + np.exp(-diff * 0.5))
        confidence_pct = float(prob_ai * 100.0)

        if confidence_pct < 50:
            label = "HUMAN"
        elif confidence_pct < 70:
            label = "SUSPICIOUS"
        else:
            label = "AI"

        return confidence_pct, label, inference_ms

    # ── Async (background-thread) inference ──────────────────────────
    def submit(self, audio: np.ndarray, source_sr: int = 48_000) -> bool:
        """Queue an audio frame for background inference. Returns False if queue full."""
        try:
            self._task_queue.put_nowait((audio, source_sr))
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.01) -> Optional[Tuple[float, str, float]]:
        """
        Non-blocking fetch of the latest inference result.
        Returns (confidence_pct, label, inference_ms) or None.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker(self):
        """Background thread that processes queued audio frames."""
        while True:
            try:
                audio, source_sr = self._task_queue.get(timeout=1.0)
                result = self.classify(audio, source_sr)
                # Only keep the latest result
                while not self._result_queue.empty():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        break
                self._result_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GMMVoiceClassifier] Worker error: {e}")
