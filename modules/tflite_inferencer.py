"""
TFLite Voice Classifier — Real-time inference engine.

Loads a quantized TFLite model and runs Mel-spectrogram-based
AI vs Human voice classification on raw audio numpy arrays.

Feature extraction parameters match the training pipeline:
  SR=16 kHz, 2.5 s windows, N_MELS=60, N_FFT=512, HOP=160, WIN=400
  → Input shape: (1, 60, 79, 1)
  → Output: probability [0=Human … 1=AI]
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
    import tensorflow as tf
except ImportError:
    raise ImportError("tensorflow is required: pip install tensorflow")


# ── Feature-extraction constants (must match training config) ────────────
SAMPLE_RATE       = 16_000
AUDIO_DURATION    = 2.5          # seconds
AUDIO_SAMPLES     = int(SAMPLE_RATE * AUDIO_DURATION)  # 40 000
N_MELS            = 60
N_FFT             = 512
HOP_LENGTH        = 160
WIN_LENGTH        = 400
FMIN              = 0
FMAX              = 8000
TARGET_FRAMES     = 79
NORM_METHOD       = "rms"
NORM_TARGET_DB    = -20.0


# ── Audio helpers ────────────────────────────────────────────────────────
def _pad_or_trim(audio: np.ndarray, target_length: int = AUDIO_SAMPLES) -> np.ndarray:
    """Pad or trim audio to target length (centre-aligned)."""
    n = len(audio)
    if n == target_length:
        return audio
    if n > target_length:
        start = (n - target_length) // 2
        return audio[start:start + target_length]
    pad_total = target_length - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode="constant")


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """RMS-normalise audio to -20 dB, matching training pipeline."""
    audio = audio - np.mean(audio)  # remove DC offset
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (NORM_TARGET_DB / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)


def _extract_mel(audio: np.ndarray) -> np.ndarray:
    """
    Extract a fixed-length Mel spectrogram → shape (60, 79, 1).
    Exactly replicates MelSpectrogramExtractor.extract() from training.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)  # log-scale

    # Fixed-length trimming / padding
    frames = mel.shape[1]
    if frames > TARGET_FRAMES:
        start = (frames - TARGET_FRAMES) // 2
        mel = mel[:, start:start + TARGET_FRAMES]
    elif frames < TARGET_FRAMES:
        pad_total = TARGET_FRAMES - frames
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        mel = np.pad(mel, ((0, 0), (pad_left, pad_right)),
                     mode="constant", constant_values=mel.min())

    # Normalise to [0, 1]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-10)

    return mel[..., np.newaxis]  # (60, 79, 1)


def silence_ratio(audio: np.ndarray, sr: int = 48_000,
                  chunk_ms: int = 50, threshold_db: float = -40.0) -> float:
    """
    Fast silence ratio: fraction of `chunk_ms`-long windows whose RMS is
    below `threshold_db` (relative to full-scale).  Pure numpy, ~µs cost.
    """
    chunk_samples = max(1, int(sr * chunk_ms / 1000))
    n_chunks = len(audio) // chunk_samples
    if n_chunks == 0:
        return 1.0
    trimmed = audio[:n_chunks * chunk_samples].reshape(n_chunks, chunk_samples)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    return float(np.mean(rms_db < threshold_db))


# ── TFLite inference wrapper ─────────────────────────────────────────────
class TFLiteVoiceClassifier:
    """
    Lightweight TFLite-based AI voice classifier.

    Call `classify(audio_array, source_sr)` for synchronous inference,
    or use `submit(audio_array, source_sr)` + `get_result()` for
    background-thread inference (non-blocking for GUIs).
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Background inference thread
        self._task_queue: queue.Queue = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Warmup with dummy data
        dummy = np.random.randn(AUDIO_SAMPLES).astype(np.float32)
        self.classify(dummy, source_sr=SAMPLE_RATE)
        print(f"[TFLiteVoiceClassifier] Loaded & warmed up: {model_path}")

    # ── Synchronous inference ────────────────────────────────────────
    def classify(
        self,
        audio: np.ndarray,
        source_sr: int = 48_000,
    ) -> Tuple[float, str, float]:
        """
        Classify an audio array.

        Args:
            audio:     1-D float32 numpy array (mono).
            source_sr: Sample rate of the input audio.

        Returns:
            (confidence_pct, label, inference_ms)
            confidence_pct: 0-100 scale (higher = more likely AI).
            label:          "HUMAN" / "SUSPICIOUS" / "AI"
            inference_ms:   wall-clock time for inference.
        """
        # Resample to 16 kHz if needed
        if source_sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=SAMPLE_RATE)

        audio = _pad_or_trim(audio)
        audio = _normalize_audio(audio)

        features = _extract_mel(audio)                         # (60, 79, 1)
        features = np.expand_dims(features, axis=0)            # (1, 60, 79, 1)
        features = features.astype(self._input_details[0]["dtype"])

        t0 = time.perf_counter()
        self._interpreter.set_tensor(self._input_details[0]["index"], features)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]["index"])
        inference_ms = (time.perf_counter() - t0) * 1000

        prob_ai = float(raw[0][0])          # 0 → Human, 1 → AI
        confidence_pct = prob_ai * 100.0

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
                print(f"[TFLiteVoiceClassifier] Worker error: {e}")
