"""
Training Pipeline with Data Augmentation
=========================================
1. Augment human voices with VoIP simulation (moderate/heavy preset)
2. Apply light augmentation to AI voices
3. Train CNN classifier on augmented data
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import augmentation from augment.py
from augment import (
    AudioAugmentor, 
    AugmentationConfig, 
    create_augmentation_preset,
    DatasetAugmentor
)

# Import classifier
from cnn_classifier import VoiceClassifier, AudioConfig


def augment_dataset(
    input_dir: str,
    output_dir: str,
    preset: str = 'moderate',
    num_augmentations: int = 3,
    max_workers: int = 4
):
    """
    Augment audio files in a directory.
    
    Args:
        input_dir: Directory with original audio files
        output_dir: Directory to save augmented files
        preset: Augmentation preset ('light', 'moderate', 'heavy', 'voip', 'cellular')
        num_augmentations: Number of augmented versions per file
        max_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print(f"Augmenting: {input_dir}")
    print(f"Preset: {preset}")
    print(f"Augmentations per file: {num_augmentations}")
    print(f"{'='*60}")
    
    # Create config from preset
    config = create_augmentation_preset(preset)
    
    # Create augmentor
    augmentor = DatasetAugmentor(config)
    
    # Process directory
    results = augmentor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        max_workers=max_workers
    )
    
    total_files = sum(len(v) for v in results.values())
    print(f"Created {total_files} augmented files from {len(results)} inputs")
    
    return results


def copy_original_files(input_dir: str, output_dir: str):
    """Copy original files to output directory (for inclusion in training)."""
    import shutil
    import librosa
    import soundfile as sf
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    files = []
    for ext in extensions:
        files.extend(input_path.glob(f"*{ext}"))
    
    print(f"Copying {len(files)} original files...")
    
    for file in tqdm(files, desc="Copying originals"):
        # Convert to WAV if needed and copy
        output_file = output_path / f"{file.stem}_original.wav"
        try:
            audio, sr = librosa.load(str(file), sr=16000, mono=True)
            sf.write(str(output_file), audio, sr)
        except Exception as e:
            print(f"Error copying {file}: {e}")


def run_augmentation_pipeline(
    ai_dir: str = "ai",
    human_dir: str = "human",
    output_base_dir: str = "augmented_data",
    human_augmentations: int = 5,
    ai_augmentations: int = 2,
    max_workers: int = 4,
    include_originals: bool = True
):
    """
    Run the full augmentation pipeline.
    
    Args:
        ai_dir: Directory with AI voice samples
        human_dir: Directory with human voice samples
        output_base_dir: Base directory for augmented output
        human_augmentations: Number of augmented versions per human file
        ai_augmentations: Number of augmented versions per AI file
        max_workers: Number of parallel workers
        include_originals: Whether to include original files in training
    """
    
    ai_output = os.path.join(output_base_dir, "ai")
    human_output = os.path.join(output_base_dir, "human")
    
    # Create output directories
    os.makedirs(ai_output, exist_ok=True)
    os.makedirs(human_output, exist_ok=True)
    
    # Step 1: Include original files if requested
    if include_originals:
        print("\n" + "="*60)
        print("STEP 1: Copying original files")
        print("="*60)
        copy_original_files(ai_dir, ai_output)
        copy_original_files(human_dir, human_output)
    
    # Step 2: Augment Human voices with VoIP simulation (moderate preset)
    print("\n" + "="*60)
    print("STEP 2: Augmenting HUMAN voices (VoIP simulation)")
    print("="*60)
    augment_dataset(
        input_dir=human_dir,
        output_dir=human_output,
        preset='moderate',  # Full VoIP simulation for human voices
        num_augmentations=human_augmentations,
        max_workers=max_workers
    )
    
    # Step 3: Augment AI voices with light preset
    print("\n" + "="*60)
    print("STEP 3: Augmenting AI voices (light augmentation)")
    print("="*60)
    augment_dataset(
        input_dir=ai_dir,
        output_dir=ai_output,
        preset='light',  # Light augmentation for AI (already VoIP-grade)
        num_augmentations=ai_augmentations,
        max_workers=max_workers
    )
    
    # Count files
    ai_files = list(Path(ai_output).glob("*.wav"))
    human_files = list(Path(human_output).glob("*.wav"))
    
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print(f"AI samples: {len(ai_files)}")
    print(f"Human samples: {len(human_files)}")
    print(f"Total: {len(ai_files) + len(human_files)}")
    
    return ai_output, human_output


def train_on_augmented_data(
    ai_dir: str,
    human_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    model_path: str = "voice_classifier_model.keras"
):
    """Train CNN on augmented data."""
    
    print("\n" + "="*60)
    print("TRAINING CNN CLASSIFIER")
    print("="*60)
    
    config = AudioConfig(
        duration=3.0,
        sample_rate=16000
    )
    
    classifier = VoiceClassifier(config)
    
    history = classifier.train(
        ai_dir=ai_dir,
        human_dir=human_dir,
        epochs=epochs,
        batch_size=batch_size,
        model_save_path=model_path
    )
    
    return classifier, history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Full training pipeline with augmentation"
    )
    
    # Data directories
    parser.add_argument("--ai-dir", default="ai", help="AI voice directory")
    parser.add_argument("--human-dir", default="human", help="Human voice directory")
    parser.add_argument("--output-dir", default="augmented_data", help="Output directory for augmented data")
    
    # Augmentation settings
    parser.add_argument("--human-aug", type=int, default=5, help="Augmentations per human file")
    parser.add_argument("--ai-aug", type=int, default=2, help="Augmentations per AI file")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--no-originals", action="store_true", help="Don't include original files")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-path", default="voice_classifier_model.keras", help="Model save path")
    
    # Mode
    parser.add_argument("--augment-only", action="store_true", help="Only run augmentation, skip training")
    parser.add_argument("--train-only", action="store_true", help="Only train (use existing augmented data)")
    
    args = parser.parse_args()
    
    if args.train_only:
        # Skip augmentation, train on existing data
        ai_output = os.path.join(args.output_dir, "ai")
        human_output = os.path.join(args.output_dir, "human")
        
        if not os.path.exists(ai_output) or not os.path.exists(human_output):
            print("Error: Augmented data not found. Run without --train-only first.")
            sys.exit(1)
    else:
        # Run augmentation
        ai_output, human_output = run_augmentation_pipeline(
            ai_dir=args.ai_dir,
            human_dir=args.human_dir,
            output_base_dir=args.output_dir,
            human_augmentations=args.human_aug,
            ai_augmentations=args.ai_aug,
            max_workers=args.workers,
            include_originals=not args.no_originals
        )
    
    if not args.augment_only:
        # Train the model
        train_on_augmented_data(
            ai_dir=ai_output,
            human_dir=human_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.model_path
        )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
