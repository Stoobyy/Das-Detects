"""
Das-Detects Modules
==================
Core modules for VoIP call detection and audio recording.
"""

from modules.check_app import is_call_active, detect_voip_sources
from modules.audio_recorder import AudioRecorder
from modules.voip_monitor import VoIPMonitor

__all__ = [
    "is_call_active",
    "detect_voip_sources", 
    "AudioRecorder",
    "VoIPMonitor",
]
