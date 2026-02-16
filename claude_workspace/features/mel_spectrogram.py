"""
Mel Spectrogram feature extraction.

Alternative to LFCC for comparison and experimentation.
"""

import numpy as np
import librosa
from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH,
    fmin: float = config.FMIN,
    fmax: Optional[float] = config.FMAX,
    power: float = 2.0,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Extract mel spectrogram from audio signal.

    Args:
        audio: Audio signal array
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length in samples
        win_length: Window length in samples
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Exponent for power spectrogram (1 for energy, 2 for power)
        log_scale: Apply log compression

    Returns:
        Mel spectrogram of shape (n_mels, time_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    if log_scale:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def extract_mel_spectrogram_fixed_length(
    audio: np.ndarray,
    target_frames: int = config.TIME_FRAMES,
    **kwargs,
) -> np.ndarray:
    """
    Extract mel spectrogram with fixed output length.

    Args:
        audio: Audio signal array
        target_frames: Target number of time frames
        **kwargs: Additional arguments for extract_mel_spectrogram

    Returns:
        Mel spectrogram of shape (n_mels, target_frames)
    """
    mel_spec = extract_mel_spectrogram(audio, **kwargs)

    current_frames = mel_spec.shape[1]

    if current_frames == target_frames:
        return mel_spec

    if current_frames > target_frames:
        # Trim from center
        start = (current_frames - target_frames) // 2
        return mel_spec[:, start:start + target_frames]

    # Pad with minimum value (for log-scale)
    min_val = mel_spec.min()
    pad_total = target_frames - current_frames
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(
        mel_spec,
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
        constant_values=min_val,
    )


class MelSpectrogramExtractor:
    """
    Mel spectrogram feature extractor class with batch processing.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        n_mels: int = config.N_MELS,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        win_length: int = config.WIN_LENGTH,
        fmin: float = config.FMIN,
        fmax: Optional[float] = config.FMAX,
        target_frames: int = config.TIME_FRAMES,
        log_scale: bool = True,
    ):
        """
        Initialize Mel spectrogram extractor.

        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length in samples
            win_length: Window length in samples
            fmin: Minimum frequency
            fmax: Maximum frequency
            target_frames: Target number of time frames
            log_scale: Apply log compression
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.target_frames = target_frames
        self.log_scale = log_scale

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio signal array

        Returns:
            Mel spectrogram of shape (n_mels, target_frames, 1)
        """
        mel_spec = extract_mel_spectrogram_fixed_length(
            audio,
            target_frames=self.target_frames,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
            log_scale=self.log_scale,
        )

        # Normalize to [0, 1] range for CNN
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-10)

        # Add channel dimension
        return mel_spec[..., np.newaxis]

    def extract_batch(self, audio_batch: list) -> np.ndarray:
        """
        Extract mel spectrogram from batch of audio signals.

        Args:
            audio_batch: List of audio signal arrays

        Returns:
            Batch of mel spectrograms of shape (batch, n_mels, target_frames, 1)
        """
        features = [self.extract(audio) for audio in audio_batch]
        return np.stack(features, axis=0)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Get output feature shape."""
        return (self.n_mels, self.target_frames, 1)
