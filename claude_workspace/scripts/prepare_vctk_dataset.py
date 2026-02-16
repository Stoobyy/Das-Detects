"""
Prepare combined VCTK + ASVspoof dataset with heavy Opus encoding.

Uses VCTK for cleaner human samples and ASVspoof for AI/spoof samples.
All samples are Opus-encoded to simulate VoIP conditions.
"""

import sys
from pathlib import Path
import numpy as np
import csv
from tqdm import tqdm
import subprocess
import tempfile
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor


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


def process_file(file_path, label, extractor, apply_opus=True):
    """Process a single audio file."""
    try:
        audio, _ = load_audio(file_path, config.SAMPLE_RATE)
        audio = pad_or_trim(audio, config.AUDIO_SAMPLES)
        audio = normalize_audio(audio)

        if apply_opus:
            bitrate = np.random.choice(["16k", "24k", "32k"])
            audio = encode_opus(audio, config.SAMPLE_RATE, bitrate)

        features = extractor.extract(audio)
        return features, label, True
    except Exception as e:
        return None, label, False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare VCTK + ASVspoof dataset")
    parser.add_argument("--vctk-dir", default=r"D:\datasets\archive (1)\VCTK-Corpus\VCTK-Corpus\human_1000")
    parser.add_argument("--asvspoof-manifest", default=r"D:\datasets\prepared\manifest_balanced.csv")
    parser.add_argument("--output", default=r"D:\datasets\prepared_vctk_opus")
    parser.add_argument("--opus-ratio", type=float, default=1.0, help="Ratio of samples to Opus encode")
    parser.add_argument("--max-human", type=int, default=None, help="Max human samples from VCTK")
    parser.add_argument("--max-ai", type=int, default=None, help="Max AI samples from ASVspoof")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = LFCCExtractor()

    # Load VCTK human samples
    vctk_dir = Path(args.vctk_dir)
    human_files = list(vctk_dir.glob("*.wav"))
    if args.max_human:
        human_files = human_files[:args.max_human]
    print(f"VCTK human samples: {len(human_files)}")

    # Load ASVspoof AI samples
    ai_files = []
    with open(args.asvspoof_manifest, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['label_int']) == 1:  # spoof/AI only
                ai_files.append(row['file_path'])

    if args.max_ai:
        ai_files = ai_files[:args.max_ai]
    print(f"ASVspoof AI samples: {len(ai_files)}")

    # Balance the dataset
    min_samples = min(len(human_files), len(ai_files))
    human_files = human_files[:min_samples]
    ai_files = ai_files[:min_samples]
    print(f"Balanced to {min_samples} samples per class")

    all_features = []
    all_labels = []
    opus_count = 0

    # Process human samples (VCTK)
    print("\nProcessing VCTK human samples...")
    for file_path in tqdm(human_files, desc="Human"):
        apply_opus = np.random.random() < args.opus_ratio
        features, label, success = process_file(str(file_path), 0, extractor, apply_opus)
        if success:
            all_features.append(features)
            all_labels.append(label)
            if apply_opus:
                opus_count += 1

    # Process AI samples (ASVspoof)
    print("\nProcessing ASVspoof AI samples...")
    for file_path in tqdm(ai_files, desc="AI"):
        apply_opus = np.random.random() < args.opus_ratio
        features, label, success = process_file(file_path, 1, extractor, apply_opus)
        if success:
            all_features.append(features)
            all_labels.append(label)
            if apply_opus:
                opus_count += 1

    # Shuffle
    indices = np.random.permutation(len(all_features))
    X = np.array(all_features)[indices]
    y = np.array(all_labels)[indices]

    print(f"\nDataset prepared:")
    print(f"  Total samples: {len(X)}")
    print(f"  Human: {sum(y == 0)}, AI: {sum(y == 1)}")
    print(f"  Opus encoded: {opus_count}")
    print(f"  Feature shape: {X.shape}")

    # Save
    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)

    import json
    metadata = {
        'num_samples': len(X),
        'human_count': int(sum(y == 0)),
        'ai_count': int(sum(y == 1)),
        'opus_count': opus_count,
        'opus_ratio': args.opus_ratio,
        'feature_shape': list(X.shape),
        'human_source': 'VCTK',
        'ai_source': 'ASVspoof',
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
