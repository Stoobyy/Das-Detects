"""
Audio utility functions for loading, resampling, and preprocessing.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Union
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = config.SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load with librosa for consistent handling
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=mono)

    return audio, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio signal array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def pad_or_trim(
    audio: np.ndarray,
    target_length: int = config.AUDIO_SAMPLES,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Pad or trim audio to target length.

    Args:
        audio: Audio signal array
        target_length: Target number of samples
        pad_mode: Numpy pad mode ('constant', 'edge', 'wrap', etc.)

    Returns:
        Audio array with target length
    """
    current_length = len(audio)

    if current_length == target_length:
        return audio

    if current_length > target_length:
        # Trim from center
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]

    # Pad symmetrically
    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(audio, (pad_left, pad_right), mode=pad_mode)


def normalize_audio(
    audio: np.ndarray,
    method: str = config.NORMALIZATION_METHOD,
    target_db: float = config.NORMALIZATION_TARGET_DB,
) -> np.ndarray:
    """
    Normalize audio signal.

    Args:
        audio: Audio signal array
        method: Normalization method ('peak', 'rms', 'none')
        target_db: Target dB level for normalization

    Returns:
        Normalized audio array
    """
    if method == "none" or len(audio) == 0:
        return audio

    # Remove DC offset
    audio = audio - np.mean(audio)

    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_amplitude = 10 ** (target_db / 20)
            audio = audio * (target_amplitude / peak)

    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)

    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def split_audio_windows(
    audio: np.ndarray,
    window_size: int = config.AUDIO_SAMPLES,
    hop_size: Optional[int] = None,
    min_length: int = None,
) -> list:
    """
    Split audio into overlapping windows.

    Args:
        audio: Audio signal array
        window_size: Size of each window in samples
        hop_size: Hop between windows (default: window_size // 2)
        min_length: Minimum length for last window (default: window_size // 2)

    Returns:
        List of audio window arrays
    """
    if hop_size is None:
        hop_size = window_size // 2

    if min_length is None:
        min_length = window_size // 2

    windows = []
    start = 0

    while start < len(audio):
        end = start + window_size

        if end <= len(audio):
            windows.append(audio[start:end])
        elif len(audio) - start >= min_length:
            # Pad last window if it's long enough
            window = pad_or_trim(audio[start:], window_size)
            windows.append(window)

        start += hop_size

    return windows


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get duration of audio file in seconds without loading full file.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    return librosa.get_duration(path=str(file_path))


def is_valid_audio(
    file_path: Union[str, Path],
    min_duration: float = config.MIN_AUDIO_LENGTH,
) -> bool:
    """
    Check if audio file is valid and meets minimum duration.

    Args:
        file_path: Path to audio file
        min_duration: Minimum duration in seconds

    Returns:
        True if audio is valid
    """
    try:
        duration = get_audio_duration(file_path)
        return duration >= min_duration
    except Exception:
        return False
