# experiments/sweep_configs.py
"""Hyperparameter sweep configurations for experiments."""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import itertools


@dataclass
class HyperparameterSweep:
    """
    Defines a sweep over hyperparameter space.

    Each key in `params` should be a dotted config path matching the
    experiment runner's config structure. For example:
        "training.learning_rate"
        "training.temperature"
        "training.mixed_precision.enabled"
        "model.embed_dim"
        "projection.hidden_dim"

    This lets the sweep system apply values directly via set_nested_config()
    without a separate mapping dict.
    """

    name: str
    params: Dict[str, List[Any]]
    description: str = ""

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        keys = list(self.params.keys())
        values = list(self.params.values())
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs


# ===== SUPERVISED BASELINE SWEEPS =====

BASELINE_LEARNING_RATE_SWEEP = HyperparameterSweep(
    name="baseline_lr_sweep",
    description="Sweep over learning rates for baseline model",
    params={
        "training.learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    },
)

BASELINE_BATCH_SIZE_SWEEP = HyperparameterSweep(
    name="baseline_batch_sweep",
    description="Sweep over batch sizes for baseline model",
    params={
        "training.batch_size": [16, 32, 64],
    },
)

BASELINE_ARCHITECTURE_SWEEP = HyperparameterSweep(
    name="baseline_arch_sweep",
    description="Sweep over model architecture parameters",
    params={
        "model.embed_dim": [128, 256, 512],
        "model.num_layers": [3, 6, 12],
        "model.heads": [4, 8, 16],
    },
)

BASELINE_DROPOUT_SWEEP = HyperparameterSweep(
    name="baseline_dropout_sweep",
    description="Sweep over dropout rates for regularization",
    params={
        "model.dropout": [0.0, 0.1, 0.2, 0.3],
    },
)

BASELINE_AUGMENTATION_SWEEP = HyperparameterSweep(
    name="baseline_augmentation_sweep",
    description="Sweep over SpecAugment configurations",
    params={
        "augmentation.prob": [0.0, 0.3, 0.5, 0.7],
        "augmentation.freq_mask_param": [3, 6, 10],
        "augmentation.time_mask_param": [5, 10, 20],
    },
)

# ===== TARGETED SUPERVISED SWEEPS =====

FOCUSED_LR_MOMENTUM_SWEEP = HyperparameterSweep(
    name="focused_lr_momentum",
    description="Fine-tune learning rate with momentum/weight decay",
    params={
        "training.learning_rate": [1e-4, 3e-4, 5e-4],
        "training.weight_decay": [0.0, 1e-5, 1e-4],
    },
)

WARMUP_SCHEDULER_SWEEP = HyperparameterSweep(
    name="warmup_scheduler",
    description="Compare different warmup and scheduling strategies",
    params={
        "training.scheduler_type": [
            "constant",
            "cosine",
            "linear_decay",
            "reduce_on_plateau",
            "cosine_warm_restarts",
        ],
        "training.warmup_steps": [0, 500, 1000],
    },
)

SCHEDULER_FINETUNE_SWEEP = HyperparameterSweep(
    name="scheduler_finetune",
    description="Fine-tune cosine scheduler parameters",
    params={
        "training.scheduler_type": ["cosine"],
        "training.warmup_steps": [250, 500, 1000, 2000],
        "training.min_lr": [1e-6, 1e-5, 1e-4],
    },
)

MIXED_PRECISION_SWEEP = HyperparameterSweep(
    name="mixed_precision_test",
    description="Test impact of mixed precision training",
    params={
        "training.mixed_precision.enabled": [False, True],
        "training.learning_rate": [1e-4, 5e-4],
    },
)

# ===== SSL SWEEPS =====

SSL_LEARNING_RATE_SWEEP = HyperparameterSweep(
    name="ssl_lr_sweep",
    description="Sweep over learning rates for SimCLR pretraining",
    params={
        "training.learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    },
)

SSL_TEMPERATURE_SWEEP = HyperparameterSweep(
    name="ssl_temperature_sweep",
    description="Sweep over contrastive temperature values",
    params={
        "training.temperature": [0.03, 0.05, 0.07, 0.1, 0.2],
    },
)

SSL_PROJECTION_SWEEP = HyperparameterSweep(
    name="ssl_projection_sweep",
    description="Sweep over projection head dimensions",
    params={
        "projection.hidden_dim": [128, 256, 512],
        "projection.output_dim": [64, 128, 256],
    },
)

SSL_BATCH_SIZE_SWEEP = HyperparameterSweep(
    name="ssl_batch_sweep",
    description="Sweep over batch sizes for SSL (larger = more negatives = better)",
    params={
        "training.batch_size": [16, 32, 64, 128],
    },
)

SSL_AUGMENTATION_SWEEP = HyperparameterSweep(
    name="ssl_augmentation_sweep",
    description="Sweep over augmentation strength for SSL views",
    params={
        "augmentation.prob": [0.3, 0.5, 0.7, 0.9],
        "augmentation.freq_mask_param": [4, 6, 10],
        "augmentation.time_mask_param": [5, 10, 20],
    },
)

SSL_WEIGHT_DECAY_SWEEP = HyperparameterSweep(
    name="ssl_weight_decay_sweep",
    description="Sweep over weight decay for SSL pretraining",
    params={
        "training.weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    },
)

SSL_QUICK_SANITY = HyperparameterSweep(
    name="ssl_quick_sanity",
    description="Quick sanity check: minimal config to verify SSL pipeline works",
    params={
        "training.learning_rate": [3e-4],
        "training.batch_size": [16],
        "training.temperature": [0.07],
    },
)

# ===== SWEEP SUITES =====

SWEEP_SUITES = {
    # --- Supervised ---
    "quick_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
    ],
    "standard_baseline": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
    ],
    "comprehensive": [
        BASELINE_LEARNING_RATE_SWEEP,
        BASELINE_BATCH_SIZE_SWEEP,
        BASELINE_ARCHITECTURE_SWEEP,
        BASELINE_DROPOUT_SWEEP,
        BASELINE_AUGMENTATION_SWEEP,
    ],
    "optimization_focus": [
        FOCUSED_LR_MOMENTUM_SWEEP,
        WARMUP_SCHEDULER_SWEEP,
        MIXED_PRECISION_SWEEP,
    ],
    "scheduler_ablation": [
        SCHEDULER_FINETUNE_SWEEP,
        WARMUP_SCHEDULER_SWEEP,
    ],
    # --- SSL ---
    "ssl_sanity": [
        SSL_QUICK_SANITY,
    ],
    "ssl_lr": [
        SSL_LEARNING_RATE_SWEEP,
    ],
    "ssl_standard": [
        SSL_LEARNING_RATE_SWEEP,
        SSL_TEMPERATURE_SWEEP,
        SSL_WEIGHT_DECAY_SWEEP,
    ],
    "ssl_comprehensive": [
        SSL_LEARNING_RATE_SWEEP,
        SSL_TEMPERATURE_SWEEP,
        SSL_PROJECTION_SWEEP,
        SSL_BATCH_SIZE_SWEEP,
        SSL_AUGMENTATION_SWEEP,
        SSL_WEIGHT_DECAY_SWEEP,
    ],
}
