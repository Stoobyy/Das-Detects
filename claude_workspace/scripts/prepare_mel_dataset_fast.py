"""
Fast parallel version of dataset preparation using MEL SPECTROGRAMS.
Identical to prepare_opus_dataset_fast.py but uses MelSpectrogramExtractor
instead of LFCCExtractor for feature comparison.
"""

import sys
from pathlib import Path
import numpy as np
import csv
from tqdm import tqdm
import subprocess
import tempfile
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from features import load_audio, pad_or_trim, normalize_audio, MelSpectrogramExtractor


def encode_opus(audio: np.ndarray, sample_rate: int = 16000, bitrate: str = "24k") -> np.ndarray:
    """Encode audio through Opus codec using ffmpeg."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            opus_path = Path(tmpdir) / "encoded.opus"
            output_path = Path(tmpdir) / "output.wav"

            audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            wavfile.write(str(input_path), sample_rate, audio_int16)

            result = subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_path),
                "-c:a", "libopus", "-b:a", bitrate,
                "-ar", str(sample_rate),
                str(opus_path)
            ], capture_output=True, timeout=10)

            if result.returncode != 0:
                return audio

            result = subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(opus_path),
                "-ar", str(sample_rate),
                str(output_path)
            ], capture_output=True, timeout=10)

            if result.returncode != 0:
                return audio

            sr, audio_out = wavfile.read(str(output_path))
            audio_out = audio_out.astype(np.float32) / 32768.0

            if len(audio_out) > len(audio):
                audio_out = audio_out[:len(audio)]
            elif len(audio_out) < len(audio):
                audio_out = np.pad(audio_out, (0, len(audio) - len(audio_out)))

            return audio_out.astype(np.float32)

    except Exception:
        return audio


def process_sample(args):
    """Process a single sample - extract MEL SPECTROGRAM features with optional Opus encoding."""
    file_path, label, apply_opus, bitrate = args

    # Create MEL extractor in each process
    extractor = MelSpectrogramExtractor()

    try:
        audio, _ = load_audio(file_path, config.SAMPLE_RATE)
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)

        # RMS normalization
        audio = normalize_audio(audio, method="rms", target_db=-20.0)

        # Opus encoding if requested
        if apply_opus:
            audio = encode_opus(audio, config.SAMPLE_RATE, bitrate)

        # Extract MEL SPECTROGRAM features
        features = extractor.extract(audio)

        return features, label, apply_opus, True
    except Exception as e:
        return None, label, apply_opus, False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset with Mel Spectrograms + Opus augmentation (FAST)")
    parser.add_argument("--manifest", default=r"D:\datasets\prepared\manifest_custom_balanced_v3.csv")
    parser.add_argument("--output", default=r"D:\datasets\prepared_mel_custom_v3_rms")
    parser.add_argument("--opus-ratio", type=float, default=0.8,
                        help="Ratio of samples to apply Opus encoding (0-1)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.workers or max(1, cpu_count() - 1)
    print(f"Using {num_workers} parallel workers")
    print(f"Feature type: MEL SPECTROGRAM (n_mels={config.N_MELS})")

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    samples = []
    with open(args.manifest, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'file_path': row['file_path'],
                'label': int(row['label_int']),
                'label_name': row['label'],
                'source': row['source']
            })

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Total samples: {len(samples)}")

    # Prepare processing tasks
    bitrates = ["16k", "24k", "32k"]
    np.random.seed(42)

    tasks = []
    for i, sample in enumerate(samples):
        apply_opus = np.random.random() < args.opus_ratio
        bitrate = np.random.choice(bitrates) if apply_opus else None
        tasks.append((sample['file_path'], sample['label'], apply_opus, bitrate))

    print(f"\nProcessing {len(tasks)} samples (Opus ratio: {args.opus_ratio})...")

    all_features = []
    all_labels = []
    opus_count = 0
    failed_count = 0

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_sample, tasks, chunksize=32),
            total=len(tasks),
            desc="Extracting Mel features"
        ))

    for features, label, applied_opus, success in results:
        if success:
            all_features.append(features)
            all_labels.append(label)
            if applied_opus:
                opus_count += 1
        else:
            failed_count += 1

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nProcessing complete:")
    print(f"  Total samples: {len(X)}")
    print(f"  With Opus encoding: {opus_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Feature shape: {X.shape}")

    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)

    metadata = {
        'num_samples': len(X),
        'opus_count': opus_count,
        'opus_ratio': args.opus_ratio,
        'feature_shape': list(X.shape),
        'source_manifest': args.manifest,
        'feature_type': 'mel_spectrogram',
    }
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  features.npy: {X.shape}")
    print(f"  labels.npy: {y.shape}")


if __name__ == "__main__":
    main()
