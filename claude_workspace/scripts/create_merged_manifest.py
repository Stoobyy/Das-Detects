"""
Create a new balanced manifest merging:
1. LibriSpeech (Human)
2. ASVspoof (AI/Spoof)
"""

import csv
import random
from pathlib import Path
import pandas as pd

def main():
    # Paths
    librispeech_manifest = Path(r"D:\datasets\prepared\manifest_librispeech.csv")
    asvspoof_manifest = Path(r"D:\datasets\prepared\manifest_balanced.csv") # We'll filter for spoof
    output_manifest = Path(r"D:\datasets\prepared\manifest_librispeech_balanced.csv")

    if not librispeech_manifest.exists():
        print(f"Error: LibriSpeech manifest not found: {librispeech_manifest}")
        return

    print("Loading manifests...")

    # Load LibriSpeech (All Human)
    df_libri = pd.read_csv(librispeech_manifest)
    df_libri['label'] = 'bonafide'
    df_libri['label_int'] = 0
    print(f"LibriSpeech samples: {len(df_libri)}")

    # Load ASVspoof (Filter for AI only)
    df_asv = pd.read_csv(asvspoof_manifest)
    df_spoof = df_asv[df_asv['label'] == 'spoof']
    print(f"ASVspoof Spoof samples: {len(df_spoof)}")

    # Balance the dataset
    n_samples = min(len(df_libri), len(df_spoof))
    print(f"\nBalancing to {n_samples} samples per class...")

    df_libri_balanced = df_libri.sample(n=n_samples, random_state=42)
    df_spoof_balanced = df_spoof.sample(n=n_samples, random_state=42)

    # Merge
    df_merged = pd.concat([df_libri_balanced, df_spoof_balanced])

    # Shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final dataset size: {len(df_merged)}")
    print(f"Human: {len(df_merged[df_merged['label'] == 'bonafide'])}")
    print(f"AI:    {len(df_merged[df_merged['label'] == 'spoof'])}")

    # Save
    df_merged.to_csv(output_manifest, index=False)
    print(f"\nSaved merged manifest to: {output_manifest}")

if __name__ == "__main__":
    main()
