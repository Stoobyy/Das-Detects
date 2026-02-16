"""Training module for model training and evaluation."""

from .trainer import Trainer, train_model
from .metrics import calculate_eer, calculate_far_frr, EERCallback

__all__ = [
    "Trainer",
    "train_model",
    "calculate_eer",
    "calculate_far_frr",
    "EERCallback",
]
