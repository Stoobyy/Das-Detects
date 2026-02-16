"""
Train model using pre-extracted features (with Opus augmentation).

Usage:
    python scripts/train_from_features.py --features D:\datasets\prepared_opus --output D:\datasets\models
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models import build_lightweight_cnn, compile_model
from training import Trainer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Limit GPU memory growth to avoid OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU config error: {e}")


def create_tf_dataset(features, labels, batch_size, shuffle=True):
    """Create TensorFlow dataset - keeps data on CPU, streams to GPU."""
    # Force CPU placement for dataset to avoid GPU OOM
    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, len(features)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train model from pre-extracted features")
    parser.add_argument("--features", required=True, help="Path to features directory")
    parser.add_argument("--output", default=r"D:\datasets\models", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()

    features_dir = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted features
    print(f"Loading features from: {features_dir}")
    X = np.load(features_dir / "features.npy")
    y = np.load(features_dir / "labels.npy")

    print(f"Loaded {len(X)} samples, shape: {X.shape}")
    print(f"Class distribution: Human={sum(y==0)}, AI={sum(y==1)}")

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=config.RANDOM_SEED, stratify=y
    )

    val_ratio = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=config.RANDOM_SEED, stratify=y_trainval
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Create datasets
    train_ds = create_tf_dataset(X_train, y_train, args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, args.batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, args.batch_size, shuffle=False)

    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    print(f"Class weights: {class_weights}")

    # Build model
    model = build_lightweight_cnn()
    compile_model(model)

    print("\nModel Summary:")
    model.summary()

    # Train
    trainer = Trainer(model=model, model_dir=str(output_dir))

    trainer.train(
        train_data=train_ds,
        val_data=val_ds,
        epochs=args.epochs,
        class_weights=class_weights,
        validation_xy=(X_val, y_val),
    )

    # Evaluate
    metrics = trainer.evaluate(test_ds)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {output_dir}")
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  EER:       {metrics['eer']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
