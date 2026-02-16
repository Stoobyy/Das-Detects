#!/usr/bin/env python3
"""
Main training script for AI voice detection model.

Usage:
    python scripts/train.py --manifest path/to/manifest.csv --epochs 50
    python scripts/train.py --data-dir path/to/data --epochs 50
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models import build_lightweight_cnn, compile_model
from training import Trainer
from training.metrics import calculate_metrics_at_threshold


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AI voice detection model"
    )

    # Data source (either manifest or data-dir)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to CSV manifest file (preferred)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (alternative to manifest)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(config.MODEL_DIR),
        help="Path to save trained models",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to use",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.INITIAL_LEARNING_RATE,
        help="Initial learning rate",
    )

    return parser.parse_args()


def train_from_manifest(
    manifest_path: str,
    model_dir: str,
    epochs: int,
    batch_size: int,
    max_samples: int = None,
):
    """Train model using manifest file."""
    from data import ManifestDatasetLoader, create_data_generators_from_manifest

    # Create data generators
    train_ds, val_ds, test_ds, info = create_data_generators_from_manifest(
        manifest_path,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    # Get validation data for EER callback
    val_x_list = []
    val_y_list = []
    for x, y in val_ds:
        val_x_list.append(x.numpy())
        val_y_list.append(y.numpy())
    X_val = np.concatenate(val_x_list)
    y_val = np.concatenate(val_y_list)

    # Build and compile model
    model = build_lightweight_cnn()
    compile_model(model)

    print("\nModel Summary:")
    model.summary()

    # Initialize trainer
    trainer = Trainer(model=model, model_dir=model_dir)

    # Train
    trainer.train(
        train_data=train_ds,
        val_data=val_ds,
        epochs=epochs,
        class_weights=info["class_weights"],
        validation_xy=(X_val, y_val),
    )

    # Evaluate
    metrics = trainer.evaluate(test_ds)

    return trainer.model, metrics


def train_from_directory(
    data_dir: str,
    model_dir: str,
    epochs: int,
    batch_size: int,
    max_samples: int = None,
):
    """Train model using directory structure."""
    from training import train_model

    return train_model(
        data_dir=data_dir,
        model_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        max_samples=max_samples,
    )


def main():
    """Main training function."""
    args = parse_args()

    # Validate input
    if not args.manifest and not args.data_dir:
        print("ERROR: Must specify either --manifest or --data-dir")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("AI VOICE DETECTION - TRAINING")
    print("=" * 60)

    if args.manifest:
        print(f"Manifest: {args.manifest}")
    else:
        print(f"Data directory: {args.data_dir}")

    print(f"Model directory: {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 60 + "\n")

    # Create model directory
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Train
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}")
            sys.exit(1)

        model, metrics = train_from_manifest(
            manifest_path=str(args.manifest),
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            sys.exit(1)

        model, metrics = train_from_directory(
            data_dir=str(args.data_dir),
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.model_dir}")
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  FAR:       {metrics['far']:.4f}")
    print(f"  FRR:       {metrics['frr']:.4f}")
    print(f"  EER:       {metrics['eer']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
