#!/usr/bin/env python3
"""
Dataset preparation script for ASVspoof data.

Parses ASVspoof metadata and creates a manifest file mapping audio files
to their labels (human/AI). Optionally creates symlinks for the training structure.

Usage:
    python scripts/prepare_dataset.py --output-dir D:\datasets\prepared
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import random


def parse_la_metadata(metadata_path: Path) -> Dict[str, str]:
    """
    Parse ASVspoof LA trial metadata.

    Format: speaker_id file_id codec channel attack_id label trim subset
    Example: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval

    Returns dict mapping file_id -> label ('bonafide' or 'spoof')
    """
    labels = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_id = parts[1]
                label = parts[5]  # 'bonafide' or 'spoof'
                labels[file_id] = label
    return labels


def parse_df_metadata(metadata_path: Path) -> Dict[str, str]:
    """
    Parse ASVspoof DF trial metadata.

    Format: speaker_id file_id codec source attack_id label trim subset vocoder_type ...
    Example: LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder

    Returns dict mapping file_id -> label
    """
    labels = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                file_id = parts[1]
                label = parts[5]  # 'bonafide' or 'spoof'
                labels[file_id] = label
    return labels


def find_audio_files(audio_dir: Path, extension: str = '.flac') -> Dict[str, Path]:
    """
    Find all audio files and return dict mapping file_id -> full path.
    """
    files = {}
    for f in audio_dir.glob(f'*{extension}'):
        file_id = f.stem  # filename without extension
        files[file_id] = f
    return files


def create_manifest(
    audio_files: Dict[str, Path],
    labels: Dict[str, str],
    output_path: Path,
) -> Tuple[int, int]:
    """
    Create manifest CSV file mapping audio paths to labels.

    Returns (human_count, ai_count)
    """
    human_count = 0
    ai_count = 0

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'label', 'label_int'])

        for file_id, audio_path in audio_files.items():
            if file_id in labels:
                label = labels[file_id]
                label_int = 0 if label == 'bonafide' else 1
                writer.writerow([str(audio_path), label, label_int])

                if label == 'bonafide':
                    human_count += 1
                else:
                    ai_count += 1

    return human_count, ai_count


def create_balanced_manifest(
    manifest_path: Path,
    output_path: Path,
    max_per_class: int = None,
    random_seed: int = 42,
):
    """
    Create a balanced manifest with equal samples per class.
    """
    random.seed(random_seed)

    human_samples = []
    ai_samples = []

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] == 'bonafide':
                human_samples.append(row)
            else:
                ai_samples.append(row)

    # Balance
    min_count = min(len(human_samples), len(ai_samples))
    if max_per_class:
        min_count = min(min_count, max_per_class)

    random.shuffle(human_samples)
    random.shuffle(ai_samples)

    balanced = human_samples[:min_count] + ai_samples[:min_count]
    random.shuffle(balanced)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'label', 'label_int'])
        writer.writeheader()
        writer.writerows(balanced)

    print(f"Created balanced manifest with {min_count} samples per class ({min_count * 2} total)")


def main():
    parser = argparse.ArgumentParser(description="Prepare ASVspoof dataset")
    parser.add_argument('--la-dir', type=str, default=r'D:\datasets\ASVspoof2021_LA_eval',
                        help='Path to ASVspoof LA eval directory')
    parser.add_argument('--df-dir', type=str, default=r'D:\datasets\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval',
                        help='Path to ASVspoof DF eval directory')
    parser.add_argument('--elevenlabs-dir', type=str,
                        default=r'D:\datasets\Fake_ElevenLabs_Respeecher\Fake_ElevenLabs_Respeecher',
                        help='Path to ElevenLabs/Respeecher samples')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for manifests')
    parser.add_argument('--balanced', action='store_true',
                        help='Create balanced dataset')
    parser.add_argument('--max-per-class', type=int, default=None,
                        help='Max samples per class for balanced dataset')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_human = 0
    total_ai = 0
    all_entries = []

    # Process LA dataset
    la_dir = Path(args.la_dir)
    if la_dir.exists():
        print(f"\nProcessing ASVspoof LA: {la_dir}")

        metadata_path = la_dir / 'LAkeys' / 'CM' / 'trial_metadata.txt'
        audio_dir = la_dir / 'flac'

        if metadata_path.exists() and audio_dir.exists():
            labels = parse_la_metadata(metadata_path)
            audio_files = find_audio_files(audio_dir, '.flac')

            for file_id, audio_path in audio_files.items():
                if file_id in labels:
                    label = labels[file_id]
                    label_int = 0 if label == 'bonafide' else 1
                    all_entries.append({
                        'file_path': str(audio_path),
                        'label': label,
                        'label_int': label_int,
                        'source': 'asvspoof_la'
                    })
                    if label == 'bonafide':
                        total_human += 1
                    else:
                        total_ai += 1

            print(f"  Found {len(audio_files)} audio files")
            print(f"  Matched {len([e for e in all_entries if e['source'] == 'asvspoof_la'])} with labels")

    # Process DF dataset
    df_dir = Path(args.df_dir)
    if df_dir.exists():
        print(f"\nProcessing ASVspoof DF: {df_dir}")

        metadata_path = df_dir / 'trial_metadata.txt'
        audio_dir = df_dir / 'flac'

        if metadata_path.exists() and audio_dir.exists():
            labels = parse_df_metadata(metadata_path)
            audio_files = find_audio_files(audio_dir, '.flac')

            for file_id, audio_path in audio_files.items():
                if file_id in labels:
                    label = labels[file_id]
                    label_int = 0 if label == 'bonafide' else 1
                    all_entries.append({
                        'file_path': str(audio_path),
                        'label': label,
                        'label_int': label_int,
                        'source': 'asvspoof_df'
                    })
                    if label == 'bonafide':
                        total_human += 1
                    else:
                        total_ai += 1

            print(f"  Found {len(audio_files)} audio files")

    # Process ElevenLabs (all AI)
    elevenlabs_dir = Path(args.elevenlabs_dir)
    if elevenlabs_dir.exists():
        print(f"\nProcessing ElevenLabs: {elevenlabs_dir}")

        for f in elevenlabs_dir.glob('*.wav'):
            all_entries.append({
                'file_path': str(f),
                'label': 'spoof',
                'label_int': 1,
                'source': 'elevenlabs'
            })
            total_ai += 1

        print(f"  Found {len([e for e in all_entries if e['source'] == 'elevenlabs'])} samples")

    # Write full manifest
    full_manifest = output_dir / 'manifest_full.csv'
    with open(full_manifest, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'label', 'label_int', 'source'])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\n{'=' * 50}")
    print(f"DATASET SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total Human (bonafide): {total_human:,}")
    print(f"Total AI (spoof):       {total_ai:,}")
    print(f"Total samples:          {total_human + total_ai:,}")
    print(f"\nFull manifest saved to: {full_manifest}")

    # Create balanced manifest if requested
    if args.balanced or args.max_per_class:
        max_samples = args.max_per_class or min(total_human, total_ai)

        random.seed(42)
        human_entries = [e for e in all_entries if e['label'] == 'bonafide']
        ai_entries = [e for e in all_entries if e['label'] == 'spoof']

        random.shuffle(human_entries)
        random.shuffle(ai_entries)

        balanced_entries = human_entries[:max_samples] + ai_entries[:max_samples]
        random.shuffle(balanced_entries)

        balanced_manifest = output_dir / 'manifest_balanced.csv'
        with open(balanced_manifest, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path', 'label', 'label_int', 'source'])
            writer.writeheader()
            writer.writerows(balanced_entries)

        print(f"\nBalanced manifest ({max_samples:,} per class): {balanced_manifest}")

    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    main()
