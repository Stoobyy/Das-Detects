import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.insert(0, str(Path(__file__).parent))
import config
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor

def analyze_audio(path, label):
    print(f"\n--- Analyzing {label}: {path.name} ---")

    # Raw load
    y, sr = librosa.load(path, sr=None)  # Load original SR
    print(f"Original SR: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f}s")
    print(f"Amplitude Range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"RMS Energy: {np.sqrt(np.mean(y**2)):.4f}")

    # Processed load (as model sees it)
    y_proc, _ = load_audio(str(path), config.SAMPLE_RATE)
    y_proc = pad_or_trim(y_proc, config.AUDIO_SAMPLES)

    # Standard Peak Normalization (Baseline)
    # y_proc = normalize_audio(y_proc, method="peak")

    # RMS Normalization (Proposed Fix)
    y_proc = normalize_audio(y_proc, method="rms", target_db=-20.0)

    # Silence check
    silence_thresh = 0.01
    silence_percent = np.mean(np.abs(y_proc) < silence_thresh) * 100
    print(f"Silence % (<{silence_thresh}): {silence_percent:.1f}%")

    # LFCC extraction
    extractor = LFCCExtractor()
    lfcc = extractor.extract(y_proc)
    print(f"LFCC Shape: {lfcc.shape}")
    print(f"LFCC Mean: {lfcc.mean():.3f}, Std: {lfcc.std():.3f}")

    return y_proc, lfcc

def plot_comparison(name1, y1, lfcc1, name2, y2, lfcc2):
    plt.figure(figsize=(12, 8))

    # Waveforms
    plt.subplot(2, 2, 1)
    librosa.display.waveshow(y1, sr=config.SAMPLE_RATE, alpha=0.7)
    plt.title(f"Waveform: {name1}")
    plt.ylim([-1, 1])

    plt.subplot(2, 2, 2)
    librosa.display.waveshow(y2, sr=config.SAMPLE_RATE, alpha=0.7, color='orange')
    plt.title(f"Waveform: {name2}")
    plt.ylim([-1, 1])

    # LFCCs
    plt.subplot(2, 2, 3)
    # Remove channel dim (60, 79, 1) -> (60, 79)
    plt.imshow(lfcc1.squeeze().T, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"LFCC: {name1}")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(lfcc2.squeeze().T, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"LFCC: {name2}")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("diagnosis_plot.png")
    print("\nSaved diagnosis_plot.png")

def main():
    # Paths
    test_path = Path(r"C:\Users\amrit\Downloads\testcases\annabel0_0001_211813_671.wav")
    # ASVspoof sample
    train_path = Path(r"D:/datasets/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/DF_E_2298812.flac")

    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}")
        return

    if not train_path.exists():
        print(f"Error: Train file not found: {train_path}")
        # Try to find another if this specific one is missing
        return

    # Analyze
    y_test, lfcc_test = analyze_audio(test_path, "TEST (Annabel)")
    y_train, lfcc_train = analyze_audio(train_path, "TRAIN (ASVspoof)")

    # Plot
    plot_comparison("TEST (Annabel)", y_test, lfcc_test, "TRAIN (ASVspoof)", y_train, lfcc_train)

if __name__ == "__main__":
    main()
