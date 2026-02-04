"""
Batch Prediction Script
=======================
Test an entire directory of audio files and get predictions.
"""

import os
import sys
from pathlib import Path
from cnn_classifier import VoiceClassifier, AudioConfig


def predict_directory(
    directory: str,
    model_path: str = "voice_classifier_model.keras",
    extensions: list = None
):
    """
    Predict all audio files in a directory.
    
    Args:
        directory: Path to directory containing audio files
        model_path: Path to trained model
        extensions: List of audio extensions to process
    """
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    
    # Load classifier
    print(f"Loading model from {model_path}...")
    classifier = VoiceClassifier()
    classifier.load_model(model_path)
    
    # Find all audio files
    dir_path = Path(directory)
    files = []
    for ext in extensions:
        files.extend(dir_path.glob(f"*{ext}"))
        files.extend(dir_path.glob(f"**/*{ext}"))  # Include subdirectories
    
    files = list(set(files))  # Remove duplicates
    files.sort()
    
    if not files:
        print(f"No audio files found in {directory}")
        return
    
    print(f"\nFound {len(files)} audio files")
    print("="*80)
    
    # Track statistics
    results = {
        'AI': [],
        'Human': [],
        'Error': []
    }
    
    # Process each file
    for file_path in files:
        try:
            label, confidence = classifier.predict(str(file_path))
            results[label].append((file_path.name, confidence))
            
            # Color-coded output
            if label == 'AI':
                status = f"🤖 AI ({confidence:.1%})"
            else:
                status = f"👤 Human ({confidence:.1%})"
            
            print(f"{status:25s} | {file_path.name}")
            
        except Exception as e:
            results['Error'].append((file_path.name, str(e)))
            print(f"❌ Error        | {file_path.name}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"🤖 AI detected:    {len(results['AI']):4d} files")
    print(f"👤 Human detected: {len(results['Human']):4d} files")
    if results['Error']:
        print(f"❌ Errors:         {len(results['Error']):4d} files")
    print(f"📁 Total:          {len(files):4d} files")
    
    # Detailed breakdown
    if results['AI']:
        print(f"\n--- AI Files (avg confidence: {sum(c for _, c in results['AI'])/len(results['AI']):.1%}) ---")
        for name, conf in sorted(results['AI'], key=lambda x: -x[1])[:10]:
            print(f"  {conf:.1%} - {name}")
        if len(results['AI']) > 10:
            print(f"  ... and {len(results['AI'])-10} more")
    
    if results['Human']:
        print(f"\n--- Human Files (avg confidence: {sum(c for _, c in results['Human'])/len(results['Human']):.1%}) ---")
        for name, conf in sorted(results['Human'], key=lambda x: -x[1])[:10]:
            print(f"  {conf:.1%} - {name}")
        if len(results['Human']) > 10:
            print(f"  ... and {len(results['Human'])-10} more")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch predict audio files")
    parser.add_argument("directory", help="Directory containing audio files")
    parser.add_argument("--model", default="voice_classifier_model.keras", help="Model path")
    
    args = parser.parse_args()
    
    predict_directory(args.directory, args.model)


if __name__ == "__main__":
    main()
