"""
Das-Detects Modules
==================
Core modules for VoIP call detection and audio recording.
"""

from modules.check_app import (
    is_call_active,
    detect_voip_sources,
    detect_any_mic_activity,
    is_call_active_strict,
    is_mic_in_use_by_voip
)
from modules.audio_recorder import AudioRecorder
from modules.voip_monitor import VoIPMonitor
from modules.gmm_inferencer import GMMVoiceClassifier
from modules.decision_engine import DecisionEngine

__all__ = [
    "is_call_active",
    "detect_voip_sources",
    "detect_any_mic_activity",
    "is_call_active_strict",
    "is_mic_in_use_by_voip",
    "AudioRecorder",
    "VoIPMonitor",
    "GMMVoiceClassifier",
    "DecisionEngine",
]
