#!/usr/bin/env python3
"""
Evaluation script for AI voice detection model.

Usage:
    python scripts/evaluate.py --model-path path/to/model.keras --data-dir path/to/data
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from training import Trainer
from training.metrics import calculate_metrics_at_threshold, find_optimal_threshold
from data import create_data_generators


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AI voice detection model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_DIR),
        help="Path to data directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (finds optimal if not specified)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for evaluation",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("AI VOICE DETECTION - EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60 + "\n")

    # Load model
    trainer = Trainer()
    trainer.load_model(args.model_path)

    # Load test data
    _, _, test_ds, info = create_data_generators(
        args.data_dir,
        batch_size=args.batch_size,
    )

    # Get predictions
    all_y_true = []
    all_y_pred = []

    print("Running inference on test set...")
    for X_batch, y_batch in test_ds:
        y_pred = trainer.model.predict(X_batch, verbose=0)
        all_y_true.extend(y_batch.numpy())
        all_y_pred.extend(y_pred.flatten())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    # Find optimal threshold if not specified
    if args.threshold is None:
        threshold, f1_score = find_optimal_threshold(y_true, y_pred, metric="f1")
        print(f"\nOptimal threshold (F1): {threshold:.3f} (F1={f1_score:.4f})")
    else:
        threshold = args.threshold

    # Calculate metrics
    metrics = calculate_metrics_at_threshold(y_true, y_pred, threshold)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Threshold: {threshold:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nError Rates:")
    print(f"  FAR (False Acceptance Rate): {metrics['far']:.4f}")
    print(f"  FRR (False Rejection Rate):  {metrics['frr']:.4f}")
    print(f"  EER (Equal Error Rate):      {metrics['eer']:.4f}")
    print(f"  EER Threshold:               {metrics['eer_threshold']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("=" * 60 + "\n")

    # Check against success criteria
    print("Success Criteria Check:")
    print(f"  [{'OK' if metrics['accuracy'] > 0.85 else 'FAIL'}] Accuracy > 85%: {metrics['accuracy']:.1%}")
    print(f"  [{'OK' if metrics['far'] < 0.10 else 'FAIL'}] FAR < 10%: {metrics['far']:.1%}")


if __name__ == "__main__":
    main()
