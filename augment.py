"""
Audio Augmentation Pipeline for Call Audio Simulation
======================================================
Transforms clean audio into realistic call-grade audio with:
- VoIP compression artifacts (G.711, Opus-like degradation)
- Network artifacts (packet loss, jitter)
- Background noise (various environments)
- Telephony bandpass filtering
- Sample rate conversion
- Clipping and saturation effects
"""

import os
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import warnings

# Audio processing
import librosa
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

# For parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation parameters."""
    
    # Sample rates
    target_sample_rates: List[int] = None  # Common telephony rates
    
    # Noise settings
    noise_types: List[str] = None
    snr_range: Tuple[float, float] = (5, 25)  # Signal-to-noise ratio in dB
    
    # VoIP/Telephony settings
    apply_bandpass: bool = True
    bandpass_low: int = 300  # Hz - telephony standard
    bandpass_high: int = 3400  # Hz - telephony standard
    
    # Compression artifacts
    apply_codec_simulation: bool = True
    codec_types: List[str] = None
    
    # Network artifacts
    apply_packet_loss: bool = True
    packet_loss_rate: Tuple[float, float] = (0.01, 0.05)  # 1-5%
    packet_size_ms: int = 20  # Typical VoIP packet size
    
    apply_jitter: bool = True
    jitter_ms: Tuple[float, float] = (5, 50)  # Jitter range in ms
    
    # Clipping/Saturation
    apply_clipping: bool = True
    clipping_threshold: Tuple[float, float] = (0.7, 0.95)
    
    # Bit depth reduction
    apply_bit_reduction: bool = True
    bit_depths: List[int] = None
    
    # Echo/Reverb
    apply_echo: bool = True
    echo_delay_ms: Tuple[float, float] = (20, 100)
    echo_decay: Tuple[float, float] = (0.1, 0.4)
    
    def __post_init__(self):
        if self.target_sample_rates is None:
            self.target_sample_rates = [8000, 16000]  # Narrowband, Wideband
        if self.noise_types is None:
            self.noise_types = ['white', 'pink', 'brown', 'office', 'street', 'cafe']
        if self.codec_types is None:
            self.codec_types = ['g711_ulaw', 'g711_alaw', 'gsm', 'opus_low', 'opus_medium']
        if self.bit_depths is None:
            self.bit_depths = [8, 12, 16]


class AudioAugmentor:
    """
    Comprehensive audio augmentation for simulating call-grade audio.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
    def augment(self, audio: np.ndarray, sr: int, 
                augmentations: Optional[List[str]] = None) -> Tuple[np.ndarray, int]:
        """
        Apply augmentations to audio.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            augmentations: List of augmentations to apply. If None, applies random subset.
        
        Returns:
            Augmented audio and new sample rate
        """
        if augmentations is None:
            augmentations = self._random_augmentation_set()
        
        augmented = audio.copy()
        current_sr = sr
        
        for aug in augmentations:
            augmented, current_sr = self._apply_augmentation(augmented, current_sr, aug)
        
        # Normalize to prevent clipping
        augmented = self._normalize(augmented)
        
        return augmented, current_sr
    
    def _random_augmentation_set(self) -> List[str]:
        """Generate a random but realistic set of augmentations."""
        all_augmentations = []
        
        # Always apply these for realistic call audio
        all_augmentations.append('resample')
        all_augmentations.append('bandpass')
        
        # Randomly add others
        if random.random() < 0.7:
            all_augmentations.append('noise')
        if random.random() < 0.6:
            all_augmentations.append('codec')
        if random.random() < 0.4:
            all_augmentations.append('packet_loss')
        if random.random() < 0.3:
            all_augmentations.append('jitter')
        if random.random() < 0.3:
            all_augmentations.append('clipping')
        if random.random() < 0.2:
            all_augmentations.append('echo')
        if random.random() < 0.4:
            all_augmentations.append('bit_reduction')
        
        return all_augmentations
    
    def _apply_augmentation(self, audio: np.ndarray, sr: int, 
                           aug_name: str) -> Tuple[np.ndarray, int]:
        """Apply a single augmentation."""
        augmentation_map = {
            'resample': self._resample,
            'bandpass': self._bandpass_filter,
            'noise': self._add_noise,
            'codec': self._simulate_codec,
            'packet_loss': self._simulate_packet_loss,
            'jitter': self._simulate_jitter,
            'clipping': self._apply_clipping,
            'echo': self._add_echo,
            'bit_reduction': self._reduce_bit_depth,
        }
        
        if aug_name in augmentation_map:
            return augmentation_map[aug_name](audio, sr)
        else:
            warnings.warn(f"Unknown augmentation: {aug_name}")
            return audio, sr
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    # ==================== AUGMENTATION METHODS ====================
    
    def _resample(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Resample to telephony sample rate."""
        target_sr = random.choice(self.config.target_sample_rates)
        if target_sr != sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    
    def _bandpass_filter(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Apply telephony bandpass filter (300Hz - 3400Hz)."""
        if not self.config.apply_bandpass:
            return audio, sr
        
        low = self.config.bandpass_low
        high = min(self.config.bandpass_high, sr // 2 - 1)  # Nyquist limit
        
        # Design Butterworth bandpass filter
        nyquist = sr / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Ensure valid frequency range
        if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1:
            return audio, sr
        
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered, sr
    
    def _add_noise(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Add various types of background noise."""
        noise_type = random.choice(self.config.noise_types)
        snr_db = random.uniform(*self.config.snr_range)
        
        noise = self._generate_noise(len(audio), sr, noise_type)
        
        # Calculate scaling factor for desired SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            noisy_audio = audio + scale * noise
        else:
            noisy_audio = audio
        
        return noisy_audio, sr
    
    def _generate_noise(self, length: int, sr: int, noise_type: str) -> np.ndarray:
        """Generate different types of noise."""
        if noise_type == 'white':
            return np.random.randn(length)
        
        elif noise_type == 'pink':
            # Pink noise: 1/f spectrum
            white = np.random.randn(length)
            # Apply 1/f filter approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            pink = signal.lfilter(b, a, white)
            return pink
        
        elif noise_type == 'brown':
            # Brown/red noise: 1/f^2 spectrum (integrated white noise)
            white = np.random.randn(length)
            brown = np.cumsum(white)
            brown = brown - np.mean(brown)
            return brown / (np.std(brown) + 1e-8)
        
        elif noise_type == 'office':
            # Office ambience: low frequency hum + occasional typing
            base_hum = np.sin(2 * np.pi * 60 * np.arange(length) / sr)  # 60Hz hum
            base_hum += 0.3 * np.sin(2 * np.pi * 120 * np.arange(length) / sr)  # Harmonic
            pink = self._generate_noise(length, sr, 'pink')
            return 0.5 * base_hum + 0.5 * pink * 0.3
        
        elif noise_type == 'street':
            # Street noise: low frequency rumble + wind
            brown = self._generate_noise(length, sr, 'brown')
            # Add some higher frequency components
            white = np.random.randn(length) * 0.1
            return brown + white
        
        elif noise_type == 'cafe':
            # Cafe babble: multiple frequency components
            noise = np.zeros(length)
            for _ in range(5):  # Multiple "voices"
                freq = random.uniform(100, 500)
                amp = random.uniform(0.1, 0.3)
                phase = random.uniform(0, 2 * np.pi)
                noise += amp * np.sin(2 * np.pi * freq * np.arange(length) / sr + phase)
            noise += 0.2 * np.random.randn(length)
            return noise
        
        else:
            return np.random.randn(length)
    
    def _simulate_codec(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate VoIP codec compression artifacts."""
        if not self.config.apply_codec_simulation:
            return audio, sr
        
        codec = random.choice(self.config.codec_types)
        
        if codec == 'g711_ulaw':
            return self._g711_ulaw(audio, sr)
        elif codec == 'g711_alaw':
            return self._g711_alaw(audio, sr)
        elif codec == 'gsm':
            return self._gsm_simulation(audio, sr)
        elif codec == 'opus_low':
            return self._opus_simulation(audio, sr, quality='low')
        elif codec == 'opus_medium':
            return self._opus_simulation(audio, sr, quality='medium')
        
        return audio, sr
    
    def _g711_ulaw(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate G.711 μ-law compression."""
        mu = 255  # μ-law parameter
        
        # Normalize to [-1, 1]
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # μ-law companding
        sign = np.sign(audio_norm)
        compressed = sign * np.log1p(mu * np.abs(audio_norm)) / np.log1p(mu)
        
        # Quantize to 8 bits
        quantized = np.round(compressed * 127) / 127
        
        # μ-law expansion
        expanded = sign * (1/mu) * ((1 + mu) ** np.abs(quantized) - 1)
        
        return expanded, sr
    
    def _g711_alaw(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate G.711 A-law compression."""
        A = 87.6  # A-law parameter
        
        # Normalize
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # A-law companding
        sign = np.sign(audio_norm)
        abs_audio = np.abs(audio_norm)
        
        compressed = np.where(
            abs_audio < 1/A,
            sign * (A * abs_audio) / (1 + np.log(A)),
            sign * (1 + np.log(A * abs_audio)) / (1 + np.log(A))
        )
        
        # Quantize to 8 bits
        quantized = np.round(compressed * 127) / 127
        
        # A-law expansion
        abs_quant = np.abs(quantized)
        threshold = 1 / (1 + np.log(A))
        
        expanded = np.where(
            abs_quant < threshold,
            np.sign(quantized) * abs_quant * (1 + np.log(A)) / A,
            np.sign(quantized) * np.exp(abs_quant * (1 + np.log(A)) - 1) / A
        )
        
        return expanded, sr
    
    def _gsm_simulation(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate GSM codec artifacts (aggressive compression)."""
        # GSM uses 13kbps, very aggressive compression
        # Simulate by heavy bandpass + quantization + spectral smoothing
        
        # Limit bandwidth (GSM nominal 300-3400Hz)
        audio, sr = self._bandpass_filter(audio, sr)
        
        # Heavy quantization (simulate low bitrate)
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        levels = 64  # Reduced levels for GSM-like quality
        quantized = np.round(audio_norm * levels) / levels
        
        # Add some "watery" artifact typical of GSM
        # Apply slight spectral smoothing
        smoothed = uniform_filter1d(quantized, size=3)
        
        return smoothed, sr
    
    def _opus_simulation(self, audio: np.ndarray, sr: int, 
                        quality: str = 'medium') -> Tuple[np.ndarray, int]:
        """Simulate Opus codec at different quality levels."""
        if quality == 'low':
            # 6-12 kbps range
            quantization_levels = 128
            smooth_window = 5
        else:  # medium
            # 16-24 kbps range
            quantization_levels = 512
            smooth_window = 3
        
        # Normalize
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Quantization
        quantized = np.round(audio_norm * quantization_levels) / quantization_levels
        
        # Slight spectral smoothing to simulate transform coding
        smoothed = uniform_filter1d(quantized, size=smooth_window)
        
        # Add subtle noise to simulate coding artifacts
        artifact_noise = np.random.randn(len(audio)) * 0.005
        result = smoothed + artifact_noise
        
        return result, sr
    
    def _simulate_packet_loss(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate VoIP packet loss."""
        if not self.config.apply_packet_loss:
            return audio, sr
        
        loss_rate = random.uniform(*self.config.packet_loss_rate)
        packet_samples = int(self.config.packet_size_ms * sr / 1000)
        
        num_packets = len(audio) // packet_samples
        result = audio.copy()
        
        for i in range(num_packets):
            if random.random() < loss_rate:
                start = i * packet_samples
                end = min(start + packet_samples, len(audio))
                
                # Different packet loss concealment strategies
                method = random.choice(['silence', 'repeat', 'interpolate'])
                
                if method == 'silence':
                    # Replace with silence (worst case)
                    result[start:end] = 0
                elif method == 'repeat':
                    # Repeat previous packet
                    if i > 0:
                        prev_start = (i - 1) * packet_samples
                        prev_end = start
                        prev_len = prev_end - prev_start
                        result[start:end] = result[prev_start:prev_start + min(prev_len, end - start)]
                elif method == 'interpolate':
                    # Linear interpolation
                    if start > 0 and end < len(audio):
                        result[start:end] = np.linspace(
                            result[start - 1], 
                            result[min(end, len(audio) - 1)],
                            end - start
                        )
        
        return result, sr
    
    def _simulate_jitter(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Simulate network jitter effects."""
        if not self.config.apply_jitter:
            return audio, sr
        
        jitter_ms = random.uniform(*self.config.jitter_ms)
        packet_samples = int(self.config.packet_size_ms * sr / 1000)
        jitter_samples = int(jitter_ms * sr / 1000)
        
        num_packets = len(audio) // packet_samples
        result = np.zeros_like(audio)
        
        for i in range(num_packets):
            # Random jitter offset
            offset = random.randint(-jitter_samples // 2, jitter_samples // 2)
            
            src_start = i * packet_samples
            src_end = min(src_start + packet_samples, len(audio))
            
            dst_start = max(0, src_start + offset)
            dst_end = min(dst_start + (src_end - src_start), len(audio))
            
            copy_len = min(src_end - src_start, dst_end - dst_start)
            if copy_len > 0:
                result[dst_start:dst_start + copy_len] = audio[src_start:src_start + copy_len]
        
        return result, sr
    
    def _apply_clipping(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Apply soft/hard clipping to simulate overdriven mic."""
        if not self.config.apply_clipping:
            return audio, sr
        
        threshold = random.uniform(*self.config.clipping_threshold)
        
        # Normalize first
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Random choice between hard and soft clipping
        if random.random() < 0.5:
            # Hard clipping
            clipped = np.clip(audio_norm, -threshold, threshold)
        else:
            # Soft clipping (tanh-based)
            clipped = np.tanh(audio_norm / threshold) * threshold
        
        return clipped, sr
    
    def _add_echo(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Add echo/reverb typical of phone calls."""
        if not self.config.apply_echo:
            return audio, sr
        
        delay_ms = random.uniform(*self.config.echo_delay_ms)
        decay = random.uniform(*self.config.echo_decay)
        
        delay_samples = int(delay_ms * sr / 1000)
        
        # Create echo
        result = audio.copy()
        if delay_samples < len(audio):
            echo = np.zeros_like(audio)
            echo[delay_samples:] = audio[:-delay_samples] * decay
            result = audio + echo
            
            # Add second, fainter echo
            if 2 * delay_samples < len(audio):
                echo2 = np.zeros_like(audio)
                echo2[2 * delay_samples:] = audio[:-2 * delay_samples] * (decay ** 2)
                result += echo2
        
        return result, sr
    
    def _reduce_bit_depth(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Reduce bit depth to simulate lower quality ADC."""
        if not self.config.apply_bit_reduction:
            return audio, sr
        
        bit_depth = random.choice(self.config.bit_depths)
        levels = 2 ** bit_depth
        
        # Normalize to [0, 1] range
        audio_norm = (audio - np.min(audio)) / (np.max(audio) - np.min(audio) + 1e-8)
        
        # Quantize
        quantized = np.round(audio_norm * (levels - 1)) / (levels - 1)
        
        # Convert back to [-1, 1] range
        result = quantized * 2 - 1
        
        return result, sr


class DatasetAugmentor:
    """
    Batch process audio files with augmentation.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.augmentor = AudioAugmentor(config)
        self.config = config or AugmentationConfig()
    
    def process_file(self, input_path: str, output_dir: str, 
                    num_augmentations: int = 5) -> List[str]:
        """
        Process a single audio file with multiple augmentation variations.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save augmented files
            num_augmentations: Number of augmented versions to create
        
        Returns:
            List of paths to augmented files
        """
        output_paths = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            
            # Get base filename
            base_name = Path(input_path).stem
            
            for i in range(num_augmentations):
                # Apply random augmentations
                augmented, new_sr = self.augmentor.augment(audio, sr)
                
                # Generate output filename
                output_name = f"{base_name}_aug_{i:03d}.wav"
                output_path = os.path.join(output_dir, output_name)
                
                # Save augmented audio
                sf.write(output_path, augmented, new_sr)
                output_paths.append(output_path)
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
        
        return output_paths
    
    def process_directory(self, input_dir: str, output_dir: str,
                         num_augmentations: int = 5,
                         extensions: List[str] = None,
                         max_workers: int = 4) -> dict:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for augmented files
            num_augmentations: Number of augmented versions per file
            extensions: List of audio file extensions to process
            max_workers: Number of parallel workers
        
        Returns:
            Dictionary mapping input files to their augmented versions
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(Path(input_dir).rglob(f"*{ext}"))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        results = {}
        
        # Process files with progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_file, 
                    str(f), 
                    output_dir, 
                    num_augmentations
                ): f for f in audio_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                input_file = futures[future]
                try:
                    output_files = future.result()
                    results[str(input_file)] = output_files
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
        
        return results


def create_augmentation_preset(preset_name: str) -> AugmentationConfig:
    """
    Create preset augmentation configurations.
    
    Presets:
        - 'light': Minimal augmentation, good audio quality
        - 'moderate': Balanced augmentation, typical call quality
        - 'heavy': Aggressive augmentation, poor call quality
        - 'voip': Focused on VoIP-specific artifacts
        - 'cellular': Focused on cellular network artifacts
    """
    if preset_name == 'light':
        return AugmentationConfig(
            target_sample_rates=[16000],
            snr_range=(20, 30),
            apply_packet_loss=False,
            apply_jitter=False,
            apply_clipping=False,
            apply_echo=False,
            bit_depths=[16]
        )
    
    elif preset_name == 'moderate':
        return AugmentationConfig(
            target_sample_rates=[8000, 16000],
            snr_range=(15, 25),
            packet_loss_rate=(0.01, 0.03),
            apply_jitter=True,
            jitter_ms=(5, 20),
            apply_clipping=True,
            clipping_threshold=(0.8, 0.95)
        )
    
    elif preset_name == 'heavy':
        return AugmentationConfig(
            target_sample_rates=[8000],
            snr_range=(5, 15),
            packet_loss_rate=(0.03, 0.1),
            apply_jitter=True,
            jitter_ms=(20, 100),
            apply_clipping=True,
            clipping_threshold=(0.6, 0.8),
            codec_types=['g711_ulaw', 'gsm'],
            bit_depths=[8, 12]
        )
    
    elif preset_name == 'voip':
        return AugmentationConfig(
            target_sample_rates=[8000, 16000],
            snr_range=(15, 25),
            codec_types=['opus_low', 'opus_medium', 'g711_ulaw', 'g711_alaw'],
            apply_packet_loss=True,
            packet_loss_rate=(0.02, 0.08),
            apply_jitter=True,
            jitter_ms=(10, 50),
            apply_echo=True,
            echo_delay_ms=(30, 80)
        )
    
    elif preset_name == 'cellular':
        return AugmentationConfig(
            target_sample_rates=[8000],
            noise_types=['street', 'white', 'pink'],
            snr_range=(10, 20),
            codec_types=['gsm', 'g711_alaw'],
            apply_packet_loss=True,
            packet_loss_rate=(0.01, 0.05),
            bandpass_low=300,
            bandpass_high=3400
        )
    
    else:
        return AugmentationConfig()


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augment audio files to simulate call-grade audio"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for augmented files"
    )
    parser.add_argument(
        "--num-augmentations", "-n",
        type=int,
        default=5,
        help="Number of augmented versions per file (default: 5)"
    )
    parser.add_argument(
        "--preset",
        choices=['light', 'moderate', 'heavy', 'voip', 'cellular'],
        default='moderate',
        help="Augmentation preset (default: moderate)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Create configuration from preset
    config = create_augmentation_preset(args.preset)
    
    # Process files
    processor = DatasetAugmentor(config)
    
    if os.path.isfile(args.input):
        # Single file
        results = processor.process_file(
            args.input, 
            args.output, 
            args.num_augmentations
        )
        print(f"Created {len(results)} augmented files")
    else:
        # Directory
        results = processor.process_directory(
            args.input,
            args.output,
            args.num_augmentations,
            max_workers=args.workers
        )
        total = sum(len(v) for v in results.values())
        print(f"Created {total} augmented files from {len(results)} inputs")
