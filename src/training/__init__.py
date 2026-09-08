# src/trainers/__init__.py
"""
Training utilities package for Bird-Intelligence-System.

Contains training loops and helpers.
"""
from .ssl_simclr_trainer import SimCLRExperimentTrainer
from .supervised_transformer_trainer import SupervisedTransformerExperimentTrainer

__all__ = [
    "SimCLRExperimentTrainer",
    "SupervisedTransformerExperimentTrainer",
]
