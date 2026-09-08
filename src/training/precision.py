# src/trainers/precision.py
"""
Precision Manager.

Provides a unified interface for:

- FP32 training
- FP16 Automatic Mixed Precision (AMP)
- BF16 Automatic Mixed Precision
- Gradient scaling
- Gradient clipping support
- Checkpoint save/load
- Training statistics

Designed so the training loop never needs to know which precision
mode is currently active.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PrecisionManager:
    """
    Handles training precision.

    Modes
    -----
    FP32
        Standard training.

    FP16
        Automatic mixed precision with GradScaler.

    BF16
        Automatic mixed precision without GradScaler.

    Parameters
    ----------
    enabled:
        Enable mixed precision.

    device:
        "cuda", "cpu", or "mps"

    use_bfloat16:
        Prefer BF16 when available.

    init_scale:
        Initial gradient scale.

    growth_interval:
        Number of successful optimizer steps before scale increases.
    """

    def __init__(
        self,
        enabled: bool = True,
        device: str = "cuda",
        use_bfloat16: bool = False,
        init_scale: float = 2.0**16,
        growth_interval: int = 2000,
    ):

        self.device = device

        self.enabled = enabled and device == "cuda" and torch.cuda.is_available()

        self.use_bfloat16 = use_bfloat16

        self.scaler: Optional[torch.amp.GradScaler] = None

        # --------------------------------------------------------------
        # FP32
        # --------------------------------------------------------------

        if not self.enabled:
            self.dtype = torch.float32
            logger.info("Precision: FP32")
            return

        # --------------------------------------------------------------
        # BF16
        # --------------------------------------------------------------

        if use_bfloat16 and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

            logger.info("Precision: BF16")

            return

        # --------------------------------------------------------------
        # FP16
        # --------------------------------------------------------------

        self.dtype = torch.float16

        self.scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=init_scale,
            growth_interval=growth_interval,
        )

        logger.info("Precision: FP16")

    # ==============================================================
    # Context manager
    # ==============================================================

    def autocast(self):
        """
        Returns autocast context.

        Usage
        -----
        with precision.autocast():
            logits = model(x)
        """

        if not self.enabled:
            return nullcontext()

        return torch.amp.autocast(
            device_type=self.device,
            dtype=self.dtype,
        )

    # ==============================================================
    # Backward
    # ==============================================================

    def scale_loss(self, loss: torch.Tensor):

        if self.scaler is None:
            return loss

        return self.scaler.scale(loss)

    # ==============================================================
    # Optimizer
    # ==============================================================

    def step(self, optimizer):

        if self.scaler is None:
            optimizer.step()
        else:
            self.scaler.step(optimizer)

    def update(self):

        if self.scaler is not None:
            self.scaler.update()

    # ==============================================================
    # Gradient utilities
    # ==============================================================

    def unscale_gradients(self, optimizer):

        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def clip_gradients(
        self,
        optimizer,
        parameters,
        max_norm: float,
    ):
        """
        AMP-safe gradient clipping.
        Returns:
            torch.Tensor
                Total gradient norm before clipping.
        """

        if max_norm is None:
            return

        self.unscale_gradients(optimizer)

        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
        )

    # ==============================================================
    # State
    # ==============================================================

    def state_dict(self):

        if self.scaler is None:
            return {}

        return self.scaler.state_dict()

    def load_state_dict(self, state):

        if self.scaler is None:
            return

        if state:
            self.scaler.load_state_dict(state)

    # ==============================================================
    # Diagnostics
    # ==============================================================

    def current_scale(self) -> float:

        if self.scaler is None:
            return 1.0

        return float(self.scaler.get_scale())

    def precision_name(self) -> str:

        if not self.enabled:
            return "fp32"

        if self.dtype == torch.bfloat16:
            return "bf16"

        return "fp16"

    def extra_state(self):

        return {
            "precision": self.precision_name(),
            "scale": self.current_scale(),
            "amp_enabled": self.enabled,
        }
