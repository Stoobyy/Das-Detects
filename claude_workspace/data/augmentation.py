"""
Audio augmentation pipeline for VoIP simulation and robustness.

Augmentations are designed to simulate real-world VoIP call conditions
and improve model generalization.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
from typing import Optional, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def add_noise(
    audio: np.ndarray,
    noise: Optional[np.ndarray] = None,
    snr_db: float = 15.0,
) -> np.ndarray:
    """
    Add noise to audio at specified SNR.

    Args:
        audio: Audio signal array
        noise: Noise signal (if None, uses white noise)
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy audio signal
    """
    if noise is None:
        noise = np.random.randn(len(audio))

    # Match noise length to audio
    if len(noise) > len(audio):
        start = np.random.randint(0, len(noise) - len(audio))
        noise = noise[start:start + len(audio)]
    elif len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))[:len(audio)]

    # Calculate power
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return audio

    # Calculate required noise power for target SNR
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    return audio + noise * noise_scale


def add_background_noise(
    audio: np.ndarray,
    snr_db: float = None,
) -> np.ndarray:
    """
    Add random background noise at random SNR.

    Args:
        audio: Audio signal array
        snr_db: SNR in dB (if None, randomly chosen)

    Returns:
        Noisy audio
    """
    if snr_db is None:
        snr_db = np.random.uniform(config.NOISE_MIN_SNR_DB, config.NOISE_MAX_SNR_DB)

    # White noise
    noise = np.random.randn(len(audio))

    return add_noise(audio, noise, snr_db)


def simulate_codec(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    bitrate: int = None,
) -> np.ndarray:
    """
    Simulate low-bitrate codec compression (VoIP-like).

    This applies a simple lowpass filter and quantization to simulate
    the effects of low-bitrate audio codecs used in VoIP.

    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        bitrate: Target bitrate (affects cutoff frequency)

    Returns:
        Codec-simulated audio
    """
    if bitrate is None:
        bitrate = np.random.randint(config.CODEC_MIN_BITRATE, config.CODEC_MAX_BITRATE)

    # Approximate cutoff frequency based on bitrate
    # Higher bitrate = higher cutoff
    cutoff_ratio = min(0.9, bitrate / 64000)
    cutoff_freq = cutoff_ratio * (sample_rate / 2)

    # Apply lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    if normalized_cutoff < 1.0:
        b, a = signal.butter(4, normalized_cutoff, btype="low")
        audio = signal.filtfilt(b, a, audio)

    # Add quantization noise
    bits = max(8, int(bitrate / 8000))
    levels = 2 ** bits
    audio = np.round(audio * levels) / levels

    return audio


def simulate_opus_codec(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    bitrate: str = None,
) -> np.ndarray:
    """
    Apply real Opus codec encoding/decoding using ffmpeg.

    This accurately simulates VoIP codecs like WhatsApp, Discord, etc.

    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        bitrate: Target bitrate string (e.g., "24k", "32k")

    Returns:
        Opus-encoded/decoded audio
    """
    import subprocess
    import tempfile

    if bitrate is None:
        # WhatsApp typically uses 24-32kbps
        bitrate = np.random.choice(["16k", "24k", "32k", "48k"])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            opus_path = Path(tmpdir) / "encoded.opus"
            output_path = Path(tmpdir) / "output.wav"

            # Write input wav
            audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            wavfile.write(str(input_path), sample_rate, audio_int16)

            # Encode to Opus
            result = subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_path),
                "-c:a", "libopus", "-b:a", bitrate,
                "-ar", str(sample_rate),
                str(opus_path)
            ], capture_output=True, timeout=10)

            if result.returncode != 0:
                return audio

            # Decode back to wav
            result = subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(opus_path),
                "-ar", str(sample_rate),
                str(output_path)
            ], capture_output=True, timeout=10)

            if result.returncode != 0:
                return audio

            # Load result
            sr, audio_out = wavfile.read(str(output_path))
            audio_out = audio_out.astype(np.float32) / 32768.0

            # Match length
            if len(audio_out) > len(audio):
                audio_out = audio_out[:len(audio)]
            elif len(audio_out) < len(audio):
                audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

            return audio_out

    except Exception:
        return audio  # Fallback to original if encoding fails


def simulate_packet_loss(
    audio: np.ndarray,
    loss_rate: float = 0.02,
    packet_size_ms: float = 20.0,
    sample_rate: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Simulate VoIP packet loss.

    Args:
        audio: Audio signal array
        loss_rate: Probability of packet loss (0-1)
        packet_size_ms: Packet size in milliseconds
        sample_rate: Sample rate

    Returns:
        Audio with simulated packet loss
    """
    packet_samples = int(packet_size_ms * sample_rate / 1000)
    num_packets = len(audio) // packet_samples

    audio = audio.copy()

    for i in range(num_packets):
        if np.random.random() < loss_rate:
            start = i * packet_samples
            end = start + packet_samples
            # Zero out lost packet (simple concealment)
            audio[start:end] = 0

    return audio


def time_shift(
    audio: np.ndarray,
    max_shift: int = config.MAX_TIME_SHIFT_SAMPLES,
) -> np.ndarray:
    """
    Randomly shift audio in time.

    Args:
        audio: Audio signal array
        max_shift: Maximum shift in samples

    Returns:
        Time-shifted audio
    """
    shift = np.random.randint(-max_shift, max_shift)

    if shift > 0:
        return np.pad(audio[:-shift], (shift, 0), mode="constant")
    elif shift < 0:
        return np.pad(audio[-shift:], (0, -shift), mode="constant")

    return audio


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    n_steps: float = None,
) -> np.ndarray:
    """
    Shift audio pitch.

    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        n_steps: Number of semitones to shift (random if None)

    Returns:
        Pitch-shifted audio
    """
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)

    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


def time_stretch(
    audio: np.ndarray,
    rate: float = None,
) -> np.ndarray:
    """
    Stretch or compress audio in time.

    Args:
        audio: Audio signal array
        rate: Stretch rate (>1 = faster, <1 = slower)

    Returns:
        Time-stretched audio
    """
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)

    return librosa.effects.time_stretch(audio, rate=rate)


def add_reverb(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    room_scale: float = None,
) -> np.ndarray:
    """
    Add simple reverb effect.

    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        room_scale: Room size (0-1)

    Returns:
        Audio with reverb
    """
    if room_scale is None:
        room_scale = np.random.uniform(0.1, 0.4)

    # Simple delay-based reverb
    delay_samples = int(room_scale * 0.05 * sample_rate)  # 5-50ms delay
    decay = 0.3 * room_scale

    if delay_samples >= len(audio):
        return audio

    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = audio[:-delay_samples] * decay

    return audio + reverb


def augment_audio(
    audio: np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    augmentations: list = None,
    probability: float = config.AUGMENT_PROBABILITY,
) -> np.ndarray:
    """
    Apply random augmentations to audio.

    Args:
        audio: Audio signal array
        sample_rate: Sample rate
        augmentations: List of augmentation functions to apply
        probability: Probability of applying each augmentation

    Returns:
        Augmented audio
    """
    if augmentations is None:
        augmentations = [
            lambda x: add_background_noise(x),
            lambda x: simulate_opus_codec(x, sample_rate),  # Use real Opus codec
            lambda x: simulate_packet_loss(x),
            lambda x: time_shift(x),
            lambda x: add_reverb(x, sample_rate),
        ]

    audio = audio.copy()

    for aug in augmentations:
        if np.random.random() < probability:
            try:
                audio = aug(audio)
            except Exception:
                pass  # Skip failed augmentation

    return audio


class AudioAugmenter:
    """
    Audio augmentation pipeline with configurable augmentations.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        probability: float = config.AUGMENT_PROBABILITY,
        enable_noise: bool = True,
        enable_codec: bool = True,
        enable_packet_loss: bool = True,
        enable_time_shift: bool = True,
        enable_reverb: bool = True,
        enable_pitch_shift: bool = False,
        enable_time_stretch: bool = False,
    ):
        """
        Initialize AudioAugmenter.

        Args:
            sample_rate: Audio sample rate
            probability: Base probability for each augmentation
            enable_*: Enable specific augmentations
        """
        self.sample_rate = sample_rate
        self.probability = probability

        self.augmentations = []

        if enable_noise:
            self.augmentations.append(("noise", add_background_noise))

        if enable_codec:
            self.augmentations.append(
                ("opus_codec", lambda x: simulate_opus_codec(x, sample_rate))
            )

        if enable_packet_loss:
            self.augmentations.append(("packet_loss", simulate_packet_loss))

        if enable_time_shift:
            self.augmentations.append(("time_shift", time_shift))

        if enable_reverb:
            self.augmentations.append(
                ("reverb", lambda x: add_reverb(x, sample_rate))
            )

        if enable_pitch_shift:
            self.augmentations.append(
                ("pitch_shift", lambda x: pitch_shift(x, sample_rate))
            )

        if enable_time_stretch:
            self.augmentations.append(("time_stretch", time_stretch))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to audio.

        Args:
            audio: Audio signal array

        Returns:
            Augmented audio
        """
        audio = audio.copy()

        for name, aug_fn in self.augmentations:
            if np.random.random() < self.probability:
                try:
                    audio = aug_fn(audio)
                except Exception:
                    pass

        return audio

    def augment_batch(self, audio_batch: list) -> list:
        """
        Augment a batch of audio signals.

        Args:
            audio_batch: List of audio arrays

        Returns:
            List of augmented audio arrays
        """
        return [self(audio) for audio in audio_batch]


class VoIPSimulator:
    """
    Simulate VoIP call conditions for realistic training data.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        noise_snr_range: tuple = (config.NOISE_MIN_SNR_DB, config.NOISE_MAX_SNR_DB),
        bitrate_range: tuple = (config.CODEC_MIN_BITRATE, config.CODEC_MAX_BITRATE),
        packet_loss_rate: float = 0.02,
    ):
        """
        Initialize VoIP simulator.

        Args:
            sample_rate: Audio sample rate
            noise_snr_range: Range of SNR values
            bitrate_range: Range of codec bitrates
            packet_loss_rate: Probability of packet loss
        """
        self.sample_rate = sample_rate
        self.noise_snr_range = noise_snr_range
        self.bitrate_range = bitrate_range
        self.packet_loss_rate = packet_loss_rate

    def simulate(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply VoIP simulation to audio.

        Args:
            audio: Audio signal array

        Returns:
            VoIP-simulated audio
        """
        # Add background noise
        snr = np.random.uniform(*self.noise_snr_range)
        audio = add_background_noise(audio, snr)

        # Apply codec simulation
        bitrate = np.random.randint(*self.bitrate_range)
        audio = simulate_codec(audio, self.sample_rate, bitrate)

        # Simulate packet loss
        audio = simulate_packet_loss(audio, self.packet_loss_rate)

        return audio
