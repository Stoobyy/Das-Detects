"""
Linear Frequency Cepstral Coefficients (LFCC) feature extraction.

LFCC uses a linear filterbank instead of mel-scale, which better captures
high-frequency artifacts common in AI-generated speech.
"""

import numpy as np
from scipy.fftpack import dct
import librosa
from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Create a linear-spaced filterbank.

    Unlike mel filterbanks, linear filterbanks have uniform spacing,
    which preserves high-frequency information important for detecting
    AI voice artifacts.

    Args:
        n_filters: Number of filters
        n_fft: FFT size
        sample_rate: Audio sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency (default: sample_rate / 2)

    Returns:
        Filterbank matrix of shape (n_filters, n_fft // 2 + 1)
    """
    if fmax is None:
        fmax = sample_rate / 2

    # Linear frequency spacing
    freqs = np.linspace(fmin, fmax, n_filters + 2)

    # Convert to FFT bin indices
    n_bins = n_fft // 2 + 1
    bins = np.floor((n_fft + 1) * freqs / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    # Create filterbank
    filterbank = np.zeros((n_filters, n_bins))

    for i in range(n_filters):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]

        # Rising slope
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left)

        # Falling slope
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center)

    return filterbank


def extract_lfcc(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    n_lfcc: int = config.N_LFCC,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WIN_LENGTH,
    n_filters: int = config.N_FILTERS,
    include_delta: bool = False,
) -> np.ndarray:
    """
    Extract LFCC features from audio signal.

    Args:
        audio: Audio signal array
        sample_rate: Audio sample rate
        n_lfcc: Number of LFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length in samples
        win_length: Window length in samples
        n_filters: Number of linear filters
        include_delta: Include delta and delta-delta features

    Returns:
        LFCC features of shape (n_lfcc, time_frames) or
        (n_lfcc * 3, time_frames) if include_delta=True
    """
    # Compute STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )
    power_spectrum = np.abs(stft) ** 2

    # Apply linear filterbank
    filterbank = linear_filterbank(n_filters, n_fft, sample_rate)
    filter_energies = np.dot(filterbank, power_spectrum)

    # Log compression
    filter_energies = np.log(filter_energies + 1e-10)

    # DCT to get cepstral coefficients
    lfcc = dct(filter_energies, type=2, axis=0, norm="ortho")[:n_lfcc]

    if include_delta:
        # Compute delta and delta-delta
        delta = librosa.feature.delta(lfcc, order=1)
        delta2 = librosa.feature.delta(lfcc, order=2)
        lfcc = np.vstack([lfcc, delta, delta2])

    return lfcc


def extract_lfcc_fixed_length(
    audio: np.ndarray,
    target_frames: int = config.TIME_FRAMES,
    **kwargs,
) -> np.ndarray:
    """
    Extract LFCC features with fixed output length.

    Args:
        audio: Audio signal array
        target_frames: Target number of time frames
        **kwargs: Additional arguments for extract_lfcc

    Returns:
        LFCC features of shape (n_lfcc, target_frames)
    """
    lfcc = extract_lfcc(audio, **kwargs)

    current_frames = lfcc.shape[1]

    if current_frames == target_frames:
        return lfcc

    if current_frames > target_frames:
        # Trim from center
        start = (current_frames - target_frames) // 2
        return lfcc[:, start:start + target_frames]

    # Pad with zeros
    pad_total = target_frames - current_frames
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(lfcc, ((0, 0), (pad_left, pad_right)), mode="constant")


class LFCCExtractor:
    """
    LFCC feature extractor class with caching and batch processing.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        n_lfcc: int = config.N_LFCC,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        win_length: int = config.WIN_LENGTH,
        n_filters: int = config.N_FILTERS,
        target_frames: int = config.TIME_FRAMES,
        include_delta: bool = False,
    ):
        """
        Initialize LFCC extractor.

        Args:
            sample_rate: Audio sample rate
            n_lfcc: Number of LFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length in samples
            win_length: Window length in samples
            n_filters: Number of linear filters
            target_frames: Target number of time frames
            include_delta: Include delta and delta-delta features
        """
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_filters = n_filters
        self.target_frames = target_frames
        self.include_delta = include_delta

        # Pre-compute filterbank
        self._filterbank = linear_filterbank(
            n_filters, n_fft, sample_rate
        )

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract LFCC features from audio.

        Args:
            audio: Audio signal array

        Returns:
            LFCC features of shape (n_lfcc, target_frames, 1)
        """
        lfcc = extract_lfcc_fixed_length(
            audio,
            target_frames=self.target_frames,
            sample_rate=self.sample_rate,
            n_lfcc=self.n_lfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_filters=self.n_filters,
            include_delta=self.include_delta,
        )

        # Add channel dimension for CNN input
        return lfcc[..., np.newaxis]

    def extract_batch(self, audio_batch: list) -> np.ndarray:
        """
        Extract LFCC features from batch of audio signals.

        Args:
            audio_batch: List of audio signal arrays

        Returns:
            Batch of LFCC features of shape (batch, n_lfcc, target_frames, 1)
        """
        features = [self.extract(audio) for audio in audio_batch]
        return np.stack(features, axis=0)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Get output feature shape."""
        n_coeffs = self.n_lfcc * 3 if self.include_delta else self.n_lfcc
        return (n_coeffs, self.target_frames, 1)
