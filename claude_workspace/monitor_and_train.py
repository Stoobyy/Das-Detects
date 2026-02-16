"""
Monitor dataset preparation and auto-start training when complete.
"""
import time
import subprocess
import sys
from pathlib import Path

output_file = Path(r"C:\Users\amrit\AppData\Local\Temp\claude\C--Users-amrit-OneDrive-Documents-GitHub-Das-Detects-claude-workspace\tasks\b836b06.output")
prepared_dir = Path(r"D:\datasets\prepared_opus")

print("Monitoring dataset preparation...")
print("Will auto-start training when complete.\n")

while True:
    # Check if output files exist (indicates completion)
    if (prepared_dir / "features.npy").exists() and (prepared_dir / "labels.npy").exists():
        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE!")
        print("=" * 60)

        # Read metadata
        import json
        meta_path = prepared_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Samples: {meta.get('num_samples', 'unknown')}")
            print(f"Opus encoded: {meta.get('opus_count', 'unknown')}")

        print("\nStarting training...")
        print("=" * 60 + "\n")

        # Start training
        result = subprocess.run([
            sys.executable,
            "scripts/train_from_features.py",
            "--features", r"D:\datasets\prepared_opus",
            "--output", r"D:\datasets\models",
            "--epochs", "50"
        ], cwd=r"C:\Users\amrit\OneDrive\Documents\GitHub\Das-Detects\claude_workspace")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!" if result.returncode == 0 else "TRAINING FAILED!")
        print("=" * 60)
        break

    # Show progress
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-3:]:
                if "Extracting features:" in line:
                    # Extract progress
                    print(f"\r{line.strip()[:80]}", end="", flush=True)
                    break
    except:
        pass

    time.sleep(30)
