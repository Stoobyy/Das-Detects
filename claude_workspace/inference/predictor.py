"""
Real-time inference for AI voice detection.

Provides both Keras and TFLite inference options.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor


class VoicePredictor:
    """
    Real-time voice detection predictor.

    Supports both Keras models (.keras) and TFLite models (.tflite).
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = config.DETECTION_THRESHOLD,
        use_tflite: bool = None,
    ):
        """
        Initialize VoicePredictor.

        Args:
            model_path: Path to model file (.keras or .tflite)
            threshold: Classification threshold
            use_tflite: Force TFLite mode (auto-detects if None)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold

        # Auto-detect model type
        if use_tflite is None:
            use_tflite = self.model_path.suffix == ".tflite"

        self.use_tflite = use_tflite

        # Load model
        if use_tflite:
            self._load_tflite_model()
        else:
            self._load_keras_model()

        # Initialize feature extractor
        self.extractor = LFCCExtractor()

        # Warmup
        self._warmup()

    def _load_keras_model(self):
        """Load Keras model."""
        from tensorflow import keras
        from models.layers import DepthwiseSeparableConv2D, ConvBlock

        self.model = keras.models.load_model(
            str(self.model_path),
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )
        print(f"Loaded Keras model from: {self.model_path}")

    def _load_tflite_model(self):
        """Load TFLite model."""
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"Loaded TFLite model from: {self.model_path}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input dtype: {self.input_details[0]['dtype']}")

    def _warmup(self):
        """Warmup model with dummy input."""
        dummy_audio = np.random.randn(config.AUDIO_SAMPLES).astype(np.float32)
        _ = self.predict_audio(dummy_audio)
        print("Model warmup complete")

    def predict_audio(self, audio: np.ndarray) -> Tuple[float, str, float]:
        """
        Predict on raw audio array.

        Args:
            audio: Audio signal array (should be 16kHz mono)

        Returns:
            Tuple of (probability, label, inference_time_ms)
        """
        # Preprocess audio
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)
        audio = normalize_audio(audio)

        # Extract features
        features = self.extractor.extract(audio)

        # Add batch dimension
        features = np.expand_dims(features, axis=0)

        # Inference
        start_time = time.perf_counter()

        if self.use_tflite:
            probability = self._predict_tflite(features)
        else:
            probability = self._predict_keras(features)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Determine label
        label = "AI-Generated" if probability >= self.threshold else "Human"

        return probability, label, inference_time

    def _predict_keras(self, features: np.ndarray) -> float:
        """Run Keras inference."""
        predictions = self.model.predict(features, verbose=0)
        return float(predictions[0][0])

    def _predict_tflite(self, features: np.ndarray) -> float:
        """Run TFLite inference."""
        # Ensure correct dtype
        input_dtype = self.input_details[0]["dtype"]
        features = features.astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], features)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return float(output[0][0])

    def predict_file(self, file_path: str) -> Tuple[float, str, float]:
        """
        Predict on audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (probability, label, inference_time_ms)
        """
        audio, _ = load_audio(file_path, config.SAMPLE_RATE)
        return self.predict_audio(audio)

    def predict_stream(
        self,
        audio_chunks: List[np.ndarray],
        aggregate: str = "mean",
    ) -> Tuple[float, str, List[float]]:
        """
        Predict on streaming audio chunks.

        Args:
            audio_chunks: List of audio arrays
            aggregate: Aggregation method ('mean', 'max', 'vote')

        Returns:
            Tuple of (final_probability, label, chunk_probabilities)
        """
        probabilities = []

        for chunk in audio_chunks:
            if len(chunk) < config.AUDIO_SAMPLES // 2:
                continue

            prob, _, _ = self.predict_audio(chunk)
            probabilities.append(prob)

        if not probabilities:
            return 0.5, "Unknown", []

        if aggregate == "mean":
            final_prob = np.mean(probabilities)
        elif aggregate == "max":
            final_prob = np.max(probabilities)
        elif aggregate == "vote":
            votes = [1 if p >= self.threshold else 0 for p in probabilities]
            final_prob = np.mean(votes)
        else:
            final_prob = np.mean(probabilities)

        label = "AI-Generated" if final_prob >= self.threshold else "Human"

        return final_prob, label, probabilities

    def benchmark(self, n_iterations: int = 100) -> dict:
        """
        Benchmark inference speed.

        Args:
            n_iterations: Number of iterations

        Returns:
            Dict with timing statistics
        """
        dummy_audio = np.random.randn(config.AUDIO_SAMPLES).astype(np.float32)
        times = []

        for _ in range(n_iterations):
            _, _, inference_time = self.predict_audio(dummy_audio)
            times.append(inference_time)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "iterations": n_iterations,
        }


def predict_audio(
    audio: Union[np.ndarray, str],
    model_path: str,
    threshold: float = config.DETECTION_THRESHOLD,
) -> Tuple[float, str]:
    """
    Simple function to predict on audio.

    Args:
        audio: Audio array or file path
        model_path: Path to model
        threshold: Classification threshold

    Returns:
        Tuple of (probability, label)
    """
    predictor = VoicePredictor(model_path, threshold)

    if isinstance(audio, str):
        prob, label, _ = predictor.predict_file(audio)
    else:
        prob, label, _ = predictor.predict_audio(audio)

    return prob, label


class StreamingPredictor:
    """
    Streaming predictor for real-time VoIP integration.

    Maintains a rolling buffer and provides predictions
    on overlapping windows.
    """

    def __init__(
        self,
        model_path: str,
        window_size: float = config.AUDIO_DURATION,
        hop_size: float = 0.5,
        threshold: float = config.DETECTION_THRESHOLD,
        smoothing_window: int = 5,
    ):
        """
        Initialize StreamingPredictor.

        Args:
            model_path: Path to model
            window_size: Analysis window in seconds
            hop_size: Hop between windows in seconds
            threshold: Classification threshold
            smoothing_window: Number of predictions to smooth over
        """
        self.predictor = VoicePredictor(model_path, threshold)
        self.window_samples = int(window_size * config.SAMPLE_RATE)
        self.hop_samples = int(hop_size * config.SAMPLE_RATE)
        self.threshold = threshold
        self.smoothing_window = smoothing_window

        # Rolling buffer
        self.buffer = np.array([], dtype=np.float32)

        # Prediction history for smoothing
        self.prediction_history = []

    def feed(self, audio_chunk: np.ndarray) -> Optional[Tuple[float, str]]:
        """
        Feed audio chunk and get prediction if window is ready.

        Args:
            audio_chunk: Audio samples to add to buffer

        Returns:
            Tuple of (probability, label) if window ready, else None
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])

        # Check if we have enough samples
        if len(self.buffer) < self.window_samples:
            return None

        # Extract window
        window = self.buffer[:self.window_samples]

        # Advance buffer
        self.buffer = self.buffer[self.hop_samples:]

        # Predict
        prob, label, _ = self.predictor.predict_audio(window)

        # Add to history
        self.prediction_history.append(prob)
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)

        # Smooth prediction
        smoothed_prob = np.mean(self.prediction_history)
        smoothed_label = "AI-Generated" if smoothed_prob >= self.threshold else "Human"

        return smoothed_prob, smoothed_label

    def reset(self):
        """Reset buffer and history."""
        self.buffer = np.array([], dtype=np.float32)
        self.prediction_history = []
