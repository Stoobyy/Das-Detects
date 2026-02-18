"""
Create balanced manifest v4 - Expanded VoIP data + 100% Opus.

Changes from v3:
  - Custom human: 1164 files (was 269)
  - Custom AI: 324 files (was 269)
  - Opus ratio: 100% (was 80%)
  - Dynamic boost factors to keep VoIP ~67% of each class
  - Base datasets (LibriSpeech + ASVspoof) always included

Composition (auto-computed, ~16k per class, ~32k total):
  Human class: custom_human(boosted) + LibriSpeech(sampled)
  AI class:    custom_ai(boosted) + ASVspoof(sampled)
"""

import csv
import random
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features import load_audio
import config


# ── Configuration ──────────────────────────────────────────────
TARGET_PER_CLASS = 16000
VOIP_TARGET_RATIO = 0.67  # Custom VoIP data should be ~67% of each class

CUSTOM_HUMAN_DIR = Path(r"C:\Users\amrit\Downloads\final_Dataset\boosting_selection\human")
CUSTOM_AI_DIR = Path(r"C:\Users\amrit\Downloads\final_Dataset\boosting_selection\ai")

LIBRISPEECH_MANIFEST = Path(r"D:\datasets\prepared\manifest_librispeech.csv")
ASVSPOOF_MANIFEST = Path(r"D:\datasets\prepared\manifest_balanced.csv")

OUTPUT_MANIFEST = Path(r"D:\datasets\prepared\manifest_custom_balanced_v5.csv")


def scan_custom_data(folder: Path, label: str, label_int: int, source: str):
    """Scan a folder for valid audio files, skipping silence/short clips."""
    samples = []

    if not folder.exists():
        print(f"Warning: Folder not found: {folder}")
        return pd.DataFrame()

    wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.flac"))
    print(f"Found {len(wav_files)} files in {folder}")

    for file_path in wav_files:
        try:
            audio, _ = load_audio(file_path, config.SAMPLE_RATE)
            if len(audio) < config.SAMPLE_RATE * 0.5:  # Skip < 0.5s
                continue
            if np.sqrt(np.mean(audio**2)) < 0.001:  # Skip silence
                continue
            samples.append({
                'file_path': str(file_path),
                'label': label,
                'label_int': label_int,
                'source': source
            })
        except Exception:
            continue

    df = pd.DataFrame(samples)
    print(f"  Valid samples: {len(df)}")
    return df


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("CREATE MERGED MANIFEST (v4 - Expanded VoIP + 100% Opus)")
    print(f"  Target per class: {TARGET_PER_CLASS}")
    print(f"  VoIP target ratio: {VOIP_TARGET_RATIO*100:.0f}%")
    print("=" * 60)

    # ── 1. Scan custom data ──
    print("\n--- Custom Human Data ---")
    custom_human = scan_custom_data(CUSTOM_HUMAN_DIR, "bonafide", 0, "custom_human")

    print("\n--- Custom AI Data ---")
    custom_ai = scan_custom_data(CUSTOM_AI_DIR, "spoof", 1, "custom_ai")

    n_custom_human = len(custom_human)
    n_custom_ai = len(custom_ai)

    # ── 2. Compute boost factors ──
    voip_target = int(TARGET_PER_CLASS * VOIP_TARGET_RATIO)

    human_boost = max(1, int(np.ceil(voip_target / n_custom_human)))
    ai_boost = max(1, int(np.ceil(voip_target / n_custom_ai)))

    boosted_human_count = n_custom_human * human_boost
    boosted_ai_count = n_custom_ai * ai_boost

    # Cap at VoIP target
    boosted_human_count = min(boosted_human_count, voip_target)
    boosted_ai_count = min(boosted_ai_count, voip_target)

    print(f"\nBoosted Custom Human: {n_custom_human} × {human_boost} = {n_custom_human * human_boost} (capped to {boosted_human_count})")
    print(f"Boosted Custom AI:    {n_custom_ai} × {ai_boost} = {n_custom_ai * ai_boost} (capped to {boosted_ai_count})")

    # ── 3. Load base datasets ──
    print("\n--- LibriSpeech (Human) ---")
    libri = pd.read_csv(LIBRISPEECH_MANIFEST)
    libri_human = libri[libri['label'] == 'bonafide'].copy()
    print(f"LibriSpeech Human samples: {len(libri_human)}")

    print("\n--- ASVspoof (AI/Spoof only) ---")
    asvspoof = pd.read_csv(ASVSPOOF_MANIFEST)
    asvspoof_spoof = asvspoof[asvspoof['label'] == 'spoof'].copy()
    print(f"ASVspoof Spoof samples: {len(asvspoof_spoof)}")

    # ── 4. Calculate base data needed ──
    base_human_needed = TARGET_PER_CLASS - boosted_human_count
    base_ai_needed = TARGET_PER_CLASS - boosted_ai_count

    print(f"\nBase Human needed (LibriSpeech): {base_human_needed}")
    print(f"Base AI needed (ASVspoof):       {base_ai_needed}")

    # ── 5. Boost custom data ──
    boosted_human_rows = pd.concat([custom_human] * human_boost, ignore_index=True)
    boosted_human_rows = boosted_human_rows.sample(n=boosted_human_count, random_state=42).reset_index(drop=True)

    boosted_ai_rows = pd.concat([custom_ai] * ai_boost, ignore_index=True)
    boosted_ai_rows = boosted_ai_rows.sample(n=boosted_ai_count, random_state=42).reset_index(drop=True)

    # ── 6. Sample base data ──
    libri_sampled = libri_human.sample(n=min(base_human_needed, len(libri_human)), random_state=42)
    asvspoof_sampled = asvspoof_spoof.sample(n=min(base_ai_needed, len(asvspoof_spoof)), random_state=42)

    # ── 7. Merge all ──
    merged = pd.concat([
        boosted_human_rows,
        boosted_ai_rows,
        libri_sampled,
        asvspoof_sampled
    ], ignore_index=True)

    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── 8. Summary ──
    total = len(merged)
    human_total = len(merged[merged['label_int'] == 0])
    ai_total = len(merged[merged['label_int'] == 1])
    voip_total = boosted_human_count + boosted_ai_count

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)
    print(f"  Total:  {total}")
    print(f"  Human:  {human_total}")
    print(f"  AI:     {ai_total}")
    print(f"\n  Source breakdown:")
    print(merged['source'].value_counts().to_string())
    print(f"\n  VoIP data ratio: {voip_total}/{total} ({voip_total/total*100:.1f}%)")
    print(f"\n  Human boost: {human_boost}x ({n_custom_human} unique clips)")
    print(f"  AI boost:    {ai_boost}x ({n_custom_ai} unique clips)")
    print("=" * 60)

    # ── 9. Save ──
    merged.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"\nSaved manifest to: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
