"""
Custom metrics for AI voice detection.

Includes Equal Error Rate (EER), False Acceptance Rate (FAR),
and False Rejection Rate (FRR).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, confusion_matrix
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras


def calculate_far_frr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR).

    FAR: Probability of accepting AI voice as human
    FRR: Probability of rejecting human voice as AI

    Args:
        y_true: Ground truth labels (0=human, 1=AI)
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Tuple of (FAR, FRR)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    # True labels
    human_mask = y_true == 0
    ai_mask = y_true == 1

    # False Acceptance: AI classified as human
    if np.sum(ai_mask) > 0:
        far = np.sum((y_pred_binary == 0) & ai_mask) / np.sum(ai_mask)
    else:
        far = 0.0

    # False Rejection: Human classified as AI
    if np.sum(human_mask) > 0:
        frr = np.sum((y_pred_binary == 1) & human_mask) / np.sum(human_mask)
    else:
        frr = 0.0

    return far, frr


def calculate_eer(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER).

    EER is the point where FAR equals FRR on the ROC curve.
    Lower EER indicates better discrimination.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities

    Returns:
        Tuple of (EER, threshold at EER)
    """
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # FRR = 1 - TPR (False Negative Rate for positive class)
    fnr = 1 - tpr

    # Find EER (where FPR = FNR)
    try:
        eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        eer_threshold = thresholds[eer_threshold_idx]
    except Exception:
        # Fallback if interpolation fails
        eer = 0.5
        eer_threshold = 0.5

    return eer, eer_threshold


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate comprehensive metrics at a given threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict with all metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    far, frr = calculate_far_frr(y_true, y_pred, threshold)
    eer, eer_threshold = calculate_eer(y_true, y_pred)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal threshold for a given metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'balanced')

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_pred, thresh)

        if metric == "f1":
            score = metrics["f1"]
        elif metric == "accuracy":
            score = metrics["accuracy"]
        elif metric == "balanced":
            # Minimize FAR + FRR
            score = 1 - (metrics["far"] + metrics["frr"]) / 2
        else:
            score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


class EERCallback(keras.callbacks.Callback):
    """
    Keras callback to calculate and log EER during training.
    """

    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        log_dir: Optional[str] = None,
    ):
        """
        Initialize EER callback.

        Args:
            validation_data: Tuple of (X_val, y_val)
            log_dir: Optional directory for TensorBoard logs
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.log_dir = log_dir
        self.eer_history = []

        if log_dir:
            self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        """Calculate EER at end of epoch."""
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred = y_pred.flatten()

        eer, eer_threshold = calculate_eer(self.y_val, y_pred)
        far, frr = calculate_far_frr(self.y_val, y_pred, eer_threshold)

        self.eer_history.append(eer)

        # Add to logs
        if logs is not None:
            logs["val_eer"] = eer
            logs["val_far"] = far
            logs["val_frr"] = frr

        print(f" - val_eer: {eer:.4f} - val_far: {far:.4f} - val_frr: {frr:.4f}")

        # TensorBoard logging
        if self.log_dir:
            with self.writer.as_default():
                tf.summary.scalar("val_eer", eer, step=epoch)
                tf.summary.scalar("val_far", far, step=epoch)
                tf.summary.scalar("val_frr", frr, step=epoch)


class DetailedMetricsCallback(keras.callbacks.Callback):
    """
    Callback to log detailed metrics including confusion matrix.
    """

    def __init__(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        threshold: float = 0.5,
        log_interval: int = 5,
    ):
        """
        Initialize callback.

        Args:
            validation_data: Tuple of (X_val, y_val)
            threshold: Classification threshold
            log_interval: Log every N epochs
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.threshold = threshold
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics periodically."""
        if (epoch + 1) % self.log_interval != 0:
            return

        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        metrics = calculate_metrics_at_threshold(self.y_val, y_pred, self.threshold)

        print(f"\n{'=' * 50}")
        print(f"Detailed Metrics (Epoch {epoch + 1})")
        print(f"{'=' * 50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"FAR: {metrics['far']:.4f}")
        print(f"FRR: {metrics['frr']:.4f}")
        print(f"EER: {metrics['eer']:.4f} @ threshold {metrics['eer_threshold']:.3f}")
        print(f"Confusion Matrix:")
        print(f"  TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}")
        print(f"  FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        print(f"{'=' * 50}\n")
