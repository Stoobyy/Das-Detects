"""
Training pipeline for AI voice detection model.

Includes training loop, callbacks, and utilities.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models import build_lightweight_cnn, compile_model
from .metrics import EERCallback, DetailedMetricsCallback, calculate_metrics_at_threshold


def create_callbacks(
    model_dir: Path,
    validation_data: Optional[Tuple] = None,
    use_eer_callback: bool = True,
    use_tensorboard: bool = True,
) -> list:
    """
    Create training callbacks.

    Args:
        model_dir: Directory to save models and logs
        validation_data: Tuple of (X_val, y_val) for EER callback
        use_eer_callback: Enable EER calculation callback
        use_tensorboard: Enable TensorBoard logging

    Returns:
        List of Keras callbacks
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    callback_list = []

    # Model checkpoint - save best model
    checkpoint_path = model_dir / "best_model.keras"
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )
    callback_list.append(checkpoint)

    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    callback_list.append(early_stop)

    # Learning rate reduction
    lr_reduce = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        min_lr=config.MIN_LEARNING_RATE,
        verbose=1,
    )
    callback_list.append(lr_reduce)

    # TensorBoard
    if use_tensorboard:
        log_dir = model_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
        )
        callback_list.append(tensorboard)

    # EER callback
    if use_eer_callback and validation_data is not None:
        eer_callback = EERCallback(validation_data)
        callback_list.append(eer_callback)

    # CSV logger
    csv_path = model_dir / "training_log.csv"
    csv_logger = callbacks.CSVLogger(str(csv_path))
    callback_list.append(csv_logger)

    return callback_list


def cosine_decay_schedule(
    initial_lr: float = config.INITIAL_LEARNING_RATE,
    min_lr: float = config.MIN_LEARNING_RATE,
    epochs: int = config.EPOCHS,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create cosine decay learning rate schedule.

    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        epochs: Total training epochs

    Returns:
        Learning rate schedule
    """
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=epochs,
        alpha=min_lr / initial_lr,
    )


class Trainer:
    """
    Training manager for AI voice detection model.
    """

    def __init__(
        self,
        model: Optional[keras.Model] = None,
        model_dir: str = None,
    ):
        """
        Initialize Trainer.

        Args:
            model: Keras model (builds new one if None)
            model_dir: Directory to save models and logs
        """
        if model is None:
            self.model = build_lightweight_cnn()
            compile_model(self.model)
        else:
            self.model = model

        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.history = None

    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        epochs: int = config.EPOCHS,
        class_weights: Optional[Dict] = None,
        callbacks_list: Optional[list] = None,
        validation_xy: Optional[Tuple] = None,
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            class_weights: Class weights for imbalanced data
            callbacks_list: Custom callbacks (uses defaults if None)
            validation_xy: Tuple of (X_val, y_val) for EER callback

        Returns:
            Training history
        """
        if callbacks_list is None:
            callbacks_list = create_callbacks(
                self.model_dir,
                validation_data=validation_xy,
            )

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model will be saved to: {self.model_dir}")

        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1,
        )

        # Save final model
        final_path = self.model_dir / "final_model.keras"
        self.model.save(str(final_path))
        print(f"\nFinal model saved to: {final_path}")

        # Save training history
        self._save_history()

        return self.history

    def _save_history(self):
        """Save training history to JSON."""
        if self.history is None:
            return

        history_path = self.model_dir / "training_history.json"

        # Convert numpy types to Python types
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

    def evaluate(
        self,
        test_data: tf.data.Dataset,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_data: Test dataset
            threshold: Classification threshold

        Returns:
            Dict with evaluation metrics
        """
        # Get predictions
        all_y_true = []
        all_y_pred = []

        for X_batch, y_batch in test_data:
            y_pred = self.model.predict(X_batch, verbose=0)
            all_y_true.extend(y_batch.numpy())
            all_y_pred.extend(y_pred.flatten())

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        # Calculate metrics
        metrics = calculate_metrics_at_threshold(y_true, y_pred, threshold)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Threshold: {threshold}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"FAR (False Acceptance Rate): {metrics['far']:.4f}")
        print(f"FRR (False Rejection Rate): {metrics['frr']:.4f}")
        print(f"EER (Equal Error Rate): {metrics['eer']:.4f}")
        print(f"EER Threshold: {metrics['eer_threshold']:.3f}")
        print("=" * 60 + "\n")

        # Save results (convert numpy types to Python types for JSON serialization)
        results_path = self.model_dir / "evaluation_results.json"
        json_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        with open(results_path, "w") as f:
            json.dump(json_metrics, f, indent=2)

        return metrics

    def save_model(self, path: Optional[str] = None):
        """Save model to file."""
        if path is None:
            path = self.model_dir / "model.keras"
        self.model.save(str(path))
        print(f"Model saved to: {path}")

    def load_model(self, path: str):
        """Load model from file."""
        from models.layers import DepthwiseSeparableConv2D, ConvBlock

        self.model = keras.models.load_model(
            path,
            custom_objects={
                "DepthwiseSeparableConv2D": DepthwiseSeparableConv2D,
                "ConvBlock": ConvBlock,
            },
        )
        print(f"Model loaded from: {path}")


def train_model(
    data_dir: str,
    model_dir: str = None,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    max_samples: Optional[int] = None,
) -> Tuple[keras.Model, Dict]:
    """
    Complete training pipeline.

    Args:
        data_dir: Path to data directory
        model_dir: Path to save models
        epochs: Number of training epochs
        batch_size: Batch size
        max_samples: Maximum samples per class

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    from data import create_data_generators

    # Create data generators
    train_ds, val_ds, test_ds, info = create_data_generators(
        data_dir,
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

    # Initialize trainer
    trainer = Trainer(model_dir=model_dir)

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
