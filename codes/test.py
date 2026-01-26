"""
Audio Augmentation Pipeline – REAL CALL MATCHED (FINAL)
=======================================================

Calibrated for:
- Windows WASAPI loopback recordings
- Desktop VoIP calls (WhatsApp / Meet / Zoom)
- Opus-like compression
- Wideband voice (≈80–7kHz)
- Mild AGC, low packet loss, no GSM harshness
"""

import os
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import warnings

import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ========================= CONFIG =========================

@dataclass
class AugmentationConfig:
    # Sample rate (matches real calls)
    target_sample_rates: List[int] = None

    # Noise (subtle)
    noise_types: List[str] = None
    snr_range: Tuple[float, float] = (18, 30)

    # Wideband VoIP bandpass
    apply_bandpass: bool = True
    bandpass_low: int = 80
    bandpass_high: int = 7000

    # Codec (Opus only)
    apply_codec_simulation: bool = True
    codec_types: List[str] = None

    # Network artifacts (rare)
    apply_packet_loss: bool = True
    packet_loss_rate: Tuple[float, float] = (0.0, 0.02)
    packet_size_ms: int = 20

    apply_jitter: bool = True
    jitter_ms: Tuple[float, float] = (0, 15)

    def __post_init__(self):
        if self.target_sample_rates is None:
            self.target_sample_rates = [16000]
        if self.noise_types is None:
            self.noise_types = ['white', 'office']
        if self.codec_types is None:
            self.codec_types = ['opus_low', 'opus_medium']


# ========================= AUGMENTOR =========================

class AudioAugmentor:
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def augment(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        augmented = audio.copy()
        current_sr = sr

        for aug in self._augmentation_chain():
            augmented, current_sr = aug(augmented, current_sr)

        augmented = self._normalize(augmented)
        return augmented, current_sr

    def _augmentation_chain(self):
        chain = [
            self._resample,
            self._bandpass_filter,
            self._simulate_codec,
            self._simulate_agc,
            self._add_noise,
        ]

        if random.random() < 0.2:
            chain.append(self._simulate_packet_loss)
        if random.random() < 0.2:
            chain.append(self._simulate_jitter)

        return chain

    # ================== CORE OPS ==================

    def _normalize(self, audio):
        peak = np.max(np.abs(audio))
        return audio / peak * 0.95 if peak > 0 else audio

    def _resample(self, audio, sr):
        target_sr = self.config.target_sample_rates[0]
        if sr != target_sr:
            audio = librosa.resample(audio, sr, target_sr)
        return audio, target_sr

    def _bandpass_filter(self, audio, sr):
        if not self.config.apply_bandpass:
            return audio, sr

        nyq = sr / 2
        low = self.config.bandpass_low / nyq
        high = min(self.config.bandpass_high / nyq, 0.99)

        if low >= high:
            return audio, sr

        b, a = signal.butter(4, [low, high], btype="band")
        return signal.filtfilt(b, a, audio), sr

    # ================== AGC (VERY IMPORTANT) ==================

    def _simulate_agc(self, audio, sr):
        frame = int(0.02 * sr)
        rms = np.sqrt(uniform_filter1d(audio ** 2, frame) + 1e-8)

        target = np.median(rms)
        gain = target / (rms + 1e-6)

        gain = uniform_filter1d(gain, frame * 5)
        gain = np.clip(gain, 0.6, 1.8)

        return audio * gain, sr

    # ================== NOISE ==================

    def _add_noise(self, audio, sr):
        snr = random.uniform(*self.config.snr_range)
        noise = np.random.randn(len(audio))

        sig_p = np.mean(audio ** 2)
        noise_p = np.mean(noise ** 2)

        scale = np.sqrt(sig_p / (noise_p * (10 ** (snr / 10))))
        return audio + noise * scale, sr

    # ================== CODEC ==================

    def _simulate_codec(self, audio, sr):
        codec = random.choice(self.config.codec_types)
        return self._opus(audio, sr, low=(codec == "opus_low"))

    def _opus(self, audio, sr, low=False):
        levels = 128 if low else 512
        smooth = 5 if low else 3

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        q = np.round(audio * levels) / levels
        q = uniform_filter1d(q, smooth)
        q += np.random.randn(len(q)) * 0.003

        return q, sr

    # ================== NETWORK ==================

    def _simulate_packet_loss(self, audio, sr):
        loss = random.uniform(*self.config.packet_loss_rate)
        pkt = int(self.config.packet_size_ms * sr / 1000)
        out = audio.copy()

        for i in range(0, len(audio), pkt):
            if random.random() < loss:
                out[i:i+pkt] = 0
        return out, sr

    def _simulate_jitter(self, audio, sr):
        jitter = int(random.uniform(*self.config.jitter_ms) * sr / 1000)
        out = np.zeros_like(audio)

        for i in range(0, len(audio), jitter + 1):
            shift = random.randint(-jitter // 2, jitter // 2)
            src = max(0, min(len(audio) - 1, i))
            dst = max(0, min(len(audio) - 1, i + shift))
            out[dst:dst+1] = audio[src:src+1]

        return out, sr


# ========================= DATASET =========================

class DatasetAugmentor:
    def __init__(self, config=None):
        self.augmentor = AudioAugmentor(config)

    def process_file(self, input_path, output_dir, n=5):
        os.makedirs(output_dir, exist_ok=True)
        audio, sr = librosa.load(input_path, sr=None, mono=True)

        base = Path(input_path).stem
        outputs = []

        for i in range(n):
            aug, new_sr = self.augmentor.augment(audio, sr)
            out = Path(output_dir) / f"{base}_aug_{i:03d}.wav"
            sf.write(out, aug, new_sr)
            outputs.append(str(out))

        return outputs


# ========================= CLI =========================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("VoIP-Matched Audio Augmentation")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-n", "--num", type=int, default=5)

    args = p.parse_args()

    aug = DatasetAugmentor(AugmentationConfig())
    aug.process_file(args.input, args.output, args.num)

    print("Done.")
