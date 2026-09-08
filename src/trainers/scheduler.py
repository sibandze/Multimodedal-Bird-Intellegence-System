# src/trainers/scheduler.py
"""
Learning rate scheduler factory.

Supports:

- Constant LR
- Linear warmup
- Cosine Annealing
- Linear Decay
- Cosine Warm Restarts
- ReduceLROnPlateau
- OneCycleLR

The scheduler factory is configuration driven and intended for
research experiments where schedulers can easily be swapped from
the configuration file.
"""

from __future__ import annotations

from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    LambdaLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
)

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_scheduler_args(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
) -> None:
    """Validate scheduler configuration."""

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")

    if min_lr < 0:
        raise ValueError("min_lr must be >= 0")

    base_lr = optimizer.param_groups[0]["lr"]

    if base_lr <= 0:
        raise ValueError("Optimizer learning rate must be > 0")

    if min_lr >= base_lr:
        raise ValueError(
            f"min_lr ({min_lr}) must be smaller than " f"optimizer lr ({base_lr})"
        )


# ---------------------------------------------------------------------
# Linear decay helper
# ---------------------------------------------------------------------


def _linear_decay_lambda(
    current_step: int,
    total_steps: int,
    min_factor: float,
):
    """
    Linear decay from 1.0 -> min_factor.

    Returns a multiplicative factor.
    """

    progress = min(current_step / total_steps, 1.0)

    return 1.0 - progress * (1.0 - min_factor)


# ---------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int = 1000,
    min_lr: float = 1e-6,
    warmup_start_factor: float = 0.1,
    plateau_factor: float = 0.5,
    plateau_patience: int = 10,
    cosine_restart_t0: Optional[int] = None,
    cosine_restart_mult: int = 2,
    one_cycle_pct_start: float = 0.3,
):
    """
    Create a learning rate scheduler.

    Parameters
    ----------
    optimizer
        Optimizer instance.

    scheduler_type
        Name of scheduler.

    warmup_steps
        Number of linear warmup iterations.

    total_steps
        Total optimizer steps.

    min_lr
        Minimum learning rate.

    Returns
    -------
    Scheduler or None.
    """

    if scheduler_type == "constant" and warmup_steps == 0:
        return None

    _validate_scheduler_args(
        optimizer,
        scheduler_type,
        warmup_steps,
        total_steps,
        min_lr,
    )

    base_lr = optimizer.param_groups[0]["lr"]

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------

    warmup_scheduler = None

    if warmup_steps > 0:

        if scheduler_type in {
            "reduce_on_plateau",
            "one_cycle",
        }:
            raise ValueError(
                f"{scheduler_type} manages its own scheduling. "
                "External warmup is not supported."
            )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    # -------------------------------------------------------------
    # Main scheduler
    # -------------------------------------------------------------

    remaining_steps = total_steps - warmup_steps

    if scheduler_type == "constant":

        main_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0,
        )

    elif scheduler_type == "cosine":

        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=remaining_steps,
            eta_min=min_lr,
        )

    elif scheduler_type == "linear_decay":

        min_factor = min_lr / base_lr

        main_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_decay_lambda(
                step,
                remaining_steps,
                min_factor,
            ),
        )

    elif scheduler_type == "cosine_warm_restarts":

        if cosine_restart_t0 is None:
            cosine_restart_t0 = max(remaining_steps // 3, 1)

        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_restart_t0,
            T_mult=cosine_restart_mult,
            eta_min=min_lr,
        )

    elif scheduler_type == "reduce_on_plateau":

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )

    elif scheduler_type == "one_cycle":

        return OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=one_cycle_pct_start,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

    else:
        raise ValueError(f"Unknown scheduler '{scheduler_type}'")

    # -------------------------------------------------------------
    # Combine warmup + scheduler
    # -------------------------------------------------------------

    if warmup_scheduler is not None:

        return SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                main_scheduler,
            ],
            milestones=[warmup_steps],
        )

    return main_scheduler


# ---------------------------------------------------------------------
# Stepping policy
# ---------------------------------------------------------------------


def get_scheduler_step_frequency(
    scheduler_type: str,
) -> str:
    """
    Returns whether scheduler should step every
    optimizer batch or every validation epoch.
    """

    if scheduler_type == "reduce_on_plateau":
        return "epoch"

    return "batch"
