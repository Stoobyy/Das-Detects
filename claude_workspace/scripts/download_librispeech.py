"""
Download and prepare LibriSpeech dataset for training.
Replacing VCTK with LibriSpeech for better generalization.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import csv
import glob
import concurrent.futures
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features import load_audio, pad_or_trim, normalize_audio

def download_file(url, destination):
    """Download file with progress bar."""
    if destination.exists():
        print(f"File already exists: {destination}")
        return

    print(f"Downloading {url} to {destination}...")
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=destination.name) as t:
        def hook(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update(b * bsize - t.n)

        urllib.request.urlretrieve(url, destination, reporthook=hook)

def extract_tar(tar_path, extract_path):
    """Extract tar.gz file with progress."""
    print(f"Extracting {tar_path}...")
    if not tar_path.exists():
        print(f"Error: Archive not found: {tar_path}")
        return

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=extract_path)

def process_file(file_path):
    """Check file validity and get metadata."""
    try:
        # Just check if we can load it
        audio, sr = load_audio(file_path, config.SAMPLE_RATE)

        # Filter very short files
        if len(audio) < config.SAMPLE_RATE * 0.5:  # < 0.5s
            return None

        return {
            'file_path': str(file_path),
            'label': 'bonafide',
            'label_int': 0,
            'source': 'librispeech'
        }
    except Exception:
        return None

def main():
    # Configuration
    dataset_url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    data_dir = Path(r"D:\datasets")
    extract_dir = data_dir / "LibriSpeech"
    archive_path = data_dir / "train-clean-100.tar.gz"
    manifest_path = data_dir / "prepared" / "manifest_librispeech.csv"

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Download
    try:
        download_file(dataset_url, archive_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # 2. Extract
    # Check if already extracted (simple check)
    if not (extract_dir / "LibriSpeech").exists():
        extract_tar(archive_path, extract_dir)
    else:
        print(f"Dataset seems to be extracted at {extract_dir}")

    # 3. Create Manifest
    print("Scanning for FLAC files...")
    # LibriSpeech structure: LibriSpeech/train-clean-100/speaker_id/chapter_id/file.flac
    search_path = extract_dir / "LibriSpeech" / "train-clean-100"
    flac_files = list(search_path.rglob("*.flac"))
    print(f"Found {len(flac_files)} files.")

    print("Verifying audio files...")
    valid_samples = []

    # Process in parallel to speed up verification
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, flac_files), total=len(flac_files)))

    valid_samples = [r for r in results if r is not None]
    print(f"Valid samples: {len(valid_samples)}")

    # 4. Save Manifest
    print(f"Saving manifest to {manifest_path}...")
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'label', 'label_int', 'source'])
        writer.writeheader()
        writer.writerows(valid_samples)

    print("\nDone! Now run prepare_opus_dataset_fast.py with this new manifest.")
    print(f"Command: python scripts/prepare_opus_dataset_fast.py --manifest \"{manifest_path}\" --output \"D:\\datasets\\prepared_opus_librispeech\"")

if __name__ == "__main__":
    main()
