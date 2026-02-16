"""Inference module for real-time AI voice detection."""

from .predictor import VoicePredictor, predict_audio
from .tflite_converter import convert_to_tflite, TFLiteConverter

__all__ = [
    "VoicePredictor",
    "predict_audio",
    "convert_to_tflite",
    "TFLiteConverter",
]
