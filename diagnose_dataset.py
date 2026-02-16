"""
Dataset Diagnostic Tool
=======================
Checks the AI vs Human voice dataset for common issues that could cause poor model performance.
"""

import os
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf

def check_audio_file(file_path):
    """Check a single audio file for issues."""
    issues = []
    try:
        # Try to load the file
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        
        # Check duration
        duration = len(y) / sr
        if duration < 0.5:
            issues.append(f"Too short: {duration:.2f}s")
        
        # Check if silent
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.001:
            issues.append(f"Nearly silent: RMS={rms:.6f}")
        
        # Check for clipping
        if np.max(np.abs(y)) > 0.99:
            issues.append("Possible clipping detected")
        
        # Check for NaN or inf
        if np.isnan(y).any() or np.isinf(y).any():
            issues.append("Contains NaN or inf values")
        
        return {
            'status': 'ok' if not issues else 'warning',
            'duration': duration,
            'sample_rate': sr,
            'rms': rms,
            'max_amplitude': float(np.max(np.abs(y))),
            'issues': issues
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def analyze_directory(directory, label, max_files=None):
    """Analyze all audio files in a directory."""
    print(f"\n{'='*80}")
    print(f"Analyzing {label} directory: {directory}")
    print(f"{'='*80}\n")
    
    dir_path = Path(directory)
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    files = []
    for ext in audio_extensions:
        files.extend(dir_path.glob(f"*{ext}"))
    
    if max_files:
        files = files[:max_files]
    
    print(f"Found {len(files)} audio files\n")
    
    if not files:
        print("❌ No audio files found!")
        return
    
    # Statistics
    stats = {
        'ok': 0,
        'warning': 0,
        'error': 0,
        'durations': [],
        'rms_values': [],
        'sample_rates': set()
    }
    
    errors = []
    warnings = []
    
    # Check each file
    for i, file_path in enumerate(files, 1):
        result = check_audio_file(file_path)
        
        if result['status'] == 'error':
            stats['error'] += 1
            errors.append((file_path.name, result['error']))
            print(f"❌ [{i:4d}] {file_path.name}: ERROR - {result['error']}")
        elif result['status'] == 'warning':
            stats['warning'] += 1
            warnings.append((file_path.name, result['issues']))
            print(f"⚠️  [{i:4d}] {file_path.name}: {', '.join(result['issues'])}")
            stats['durations'].append(result['duration'])
            stats['rms_values'].append(result['rms'])
            stats['sample_rates'].add(result['sample_rate'])
        else:
            stats['ok'] += 1
            stats['durations'].append(result['duration'])
            stats['rms_values'].append(result['rms'])
            stats['sample_rates'].add(result['sample_rate'])
            if i <= 10:  # Show first 10 ok files
                print(f"✓  [{i:4d}] {file_path.name}: {result['duration']:.2f}s, RMS={result['rms']:.4f}")
    
    # Summary
    print(f"\n{'-'*80}")
    print(f"SUMMARY FOR {label}")
    print(f"{'-'*80}")
    print(f"✓  OK:       {stats['ok']:4d} files")
    print(f"⚠️  Warnings: {stats['warning']:4d} files")
    print(f"❌ Errors:   {stats['error']:4d} files")
    print(f"📁 Total:    {len(files):4d} files")
    
    if stats['durations']:
        print(f"\nAudio Statistics:")
        print(f"  Duration: {np.mean(stats['durations']):.2f}s ± {np.std(stats['durations']):.2f}s")
        print(f"            (min: {np.min(stats['durations']):.2f}s, max: {np.max(stats['durations']):.2f}s)")
        print(f"  RMS:      {np.mean(stats['rms_values']):.4f} ± {np.std(stats['rms_values']):.4f}")
        print(f"  Sample Rates: {sorted(stats['sample_rates'])}")
    
    # Show top errors
    if errors:
        print(f"\n⚠️  TOP ERRORS:")
        for name, error in errors[:10]:
            print(f"  - {name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")
    
    # Show top warnings
    if warnings:
        print(f"\n⚠️  TOP WARNINGS:")
        for name, issues in warnings[:10]:
            print(f"  - {name}: {', '.join(issues)}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings)-10} more")
    
    return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose dataset issues")
    parser.add_argument("--ai-dir", "-a", default="ai", help="AI samples directory")
    parser.add_argument("--human-dir", "-u", default="human", help="Human samples directory")
    parser.add_argument("--max-files", "-m", type=int, default=None, help="Max files to check per directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DATASET DIAGNOSTIC TOOL")
    print("="*80)
    
    # Analyze both directories
    ai_stats = analyze_directory(args.ai_dir, "AI", args.max_files)
    human_stats = analyze_directory(args.human_dir, "HUMAN", args.max_files)
    
    # Overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON")
    print(f"{'='*80}")
    
    if ai_stats and human_stats:
        if ai_stats['durations'] and human_stats['durations']:
            print(f"\nAverage Durations:")
            print(f"  AI:    {np.mean(ai_stats['durations']):.2f}s ± {np.std(ai_stats['durations']):.2f}s")
            print(f"  Human: {np.mean(human_stats['durations']):.2f}s ± {np.std(human_stats['durations']):.2f}s")
            
            print(f"\nAverage RMS (loudness):")
            print(f"  AI:    {np.mean(ai_stats['rms_values']):.4f} ± {np.std(ai_stats['rms_values']):.4f}")
            print(f"  Human: {np.mean(human_stats['rms_values']):.4f} ± {np.std(human_stats['rms_values']):.4f}")
            
            # Check for significant differences
            ai_rms_mean = np.mean(ai_stats['rms_values'])
            human_rms_mean = np.mean(human_stats['rms_values'])
            if abs(ai_rms_mean - human_rms_mean) > 0.05:
                print(f"\n⚠️  WARNING: Significant RMS difference detected!")
                print(f"   This could indicate volume/quality mismatch between classes.")
    
    print("\n" + "="*80)
    print("Diagnosis complete! Review the warnings and errors above.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
