# src/trainers/base_trainer.py

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .precision import PrecisionManager
from .scheduler import create_scheduler, get_scheduler_step_frequency
from src.utils.memory_utils import get_gpu_memory_info
from .callbacks import Callback, CallbackRunner


class BaseTrainer(ABC):
    """
    Abstract base trainer implementing shared training infrastructure.

    Subclasses must set ``best_monitor`` and ``best_mode`` *before* calling
    ``super().__init__()``, then implement the five abstract methods.

    Abstract hooks
    --------------
    _get_default_callbacks  – callback list for this trainer type
    get_dataloaders         – build data loaders (must set self.train_loader / self.val_loader)
    _build_model            – construct and return the model
    _train_epoch            – one training epoch → dict of train metrics
    _validate_epoch         – one validation epoch → dict of val metrics

    Overridable hooks (sensible defaults)
    --------------------------------------
    _on_pre_train           – runs after checkpoint resume, before the epoch loop
    _restore_best_from_state– restore subclass-specific best-tracking keys from a checkpoint
    _post_train             – post-training work (e.g. test evaluation) → result dict
    _print_train_init       – startup banner
    _print_epoch_summary    – one-line per-epoch log
    """

    _weight_decay_default: float = 1e-4

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(exist_ok=True, parents=True)

        # ── device & precision ──────────────────────────────────────────
        self.device = torch.device(
            config["training"].get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.precision = PrecisionManager(
            enabled=config["training"].get("mixed_precision", {}).get("enabled", True),
            device=self.device.type,
            use_bfloat16=config["training"]
            .get("mixed_precision", {})
            .get("use_bfloat16", False),
        )

        # ── git hash (for checkpoint provenance) ────────────────────────
        try:
            import subprocess

            self.git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception:
            self.git_hash = None

        # ── best-metric tracking ────────────────────────────────────────
        # Subclasses must set best_monitor / best_mode before super().__init__().
        if not hasattr(self, "best_monitor"):
            self.best_monitor = "val_loss"
        if not hasattr(self, "best_mode"):
            self.best_mode = "min"

        self.best_metric_value = float("inf") if self.best_mode == "min" else 0.0
        self.best_epoch = 0
        self.trainer_state: Dict[str, Any] = {
            "best_metric": self.best_metric_value,
            "best_epoch": self.best_epoch,
        }
        self.stop_training = False

        # ── callbacks ───────────────────────────────────────────────────
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        self.cb_runner = CallbackRunner(callbacks)

    # ==================================================================
    # Public API
    # ==================================================================

    def request_stop(self):
        """Clean external API for callbacks to trigger early stopping."""
        self.stop_training = True

    def train(self, df) -> Dict[str, Any]:
        """Main training loop — template method calling subclass hooks."""
        # 1. Data
        self.get_dataloaders(df)

        # 2. Model
        self.model = self._build_model()
        self._print_train_init()

        # 3. Optimizer & scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # 4. Resume
        resume_epoch = self._resume_checkpoint()

        # 5. Pre-train hook (compile, save label maps, …)
        self._on_pre_train()

        # 6. Callbacks — train begin
        self.cb_runner.on_train_begin(self)

        # 7. Epoch loop
        epochs = self.config["training"]["epochs"]
        self._last_train_metrics: Dict[str, float] = {}
        self._last_val_metrics: Dict[str, float] = {}

        for epoch in range(resume_epoch, epochs):
            if self.stop_training:
                break

            self.train_loader.dataset.set_epoch(epoch)
            self.cb_runner.on_epoch_begin(self, epoch)
            epoch_start = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._update_best(val_metrics, epoch)
            self._step_scheduler(val_metrics)

            self._last_train_metrics = train_metrics
            self._last_val_metrics = val_metrics

            epoch_duration = time.time() - epoch_start
            logs = self._build_logs(epoch, train_metrics, val_metrics, epoch_duration)
            self._print_epoch_summary(logs)
            self.cb_runner.on_epoch_end(self, epoch, logs)

        # 8. Post-train & cleanup
        results = self._post_train()
        self.cb_runner.on_train_end(self)
        return results

    # ==================================================================
    # Abstract — subclasses MUST implement
    # ==================================================================

    @abstractmethod
    def _get_default_callbacks(self) -> List[Callback]:
        """Return the default callback list for this trainer type."""
        ...

    @abstractmethod
    def get_dataloaders(self, df):
        """Build data loaders.  Must set at least self.train_loader and
        self.val_loader."""
        ...

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Construct and return the model (already on self.device)."""
        ...

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.

        Available instance attrs: train_loader, model, optimizer,
        precision, device, scheduler, _scheduler_step_freq.

        Returns a dict with at least ``train_loss``.
        May include ``avg_grad_norm`` and ``train_total`` for logging.
        When ``train_total`` is present, ``samples_per_sec`` is
        computed automatically in ``_build_logs``.
        """
        ...

    @abstractmethod
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one validation epoch.

        Returns a dict with validation metrics (e.g. ``val_loss``, ``val_acc``).
        Return zeros / empty dict if no validation data is available.
        """
        ...

    # ==================================================================
    # Overridable hooks — sensible defaults
    # ==================================================================

    def _on_pre_train(self):
        """Called after model/scheduler/checkpoint setup, before the epoch
        loop.  Use for compilation, saving label maps, etc."""

    def _restore_best_from_state(self, state: Dict):
        """Hook for subclasses to restore additional best-tracking keys
        from a checkpoint's trainer_state dict (e.g. legacy key names).

        The default implementation handles the common case of loading
        old checkpoints that stored ``best_val_acc`` or ``best_loss``
        instead of the current ``best_metric`` key."""

    def _post_train(self) -> Dict[str, Any]:
        """Post-training work (e.g. test evaluation).  Returns the final
        result dict that ``train()`` will return."""
        return {}

    def _print_train_init(self):
        """Startup banner.  Override to append subclass-specific info."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"\n>>> Initializing {self.__class__.__name__}:")
        print(f"    Device:    {self.device}")
        print(f"    Precision: {self.precision.precision_name()}")
        print(f"    Params:    {num_params:,} (Trainable: {trainable_params:,})")
        print(f"    Train samples: {len(self.train_loader.dataset)}")
        if getattr(self, "val_loader", None) is not None:
            print(f"    Val samples:   {len(self.val_loader.dataset)}")

    def _print_epoch_summary(self, logs: Dict[str, Any]):
        """One-line epoch log.  Override for custom formatting."""
        print(
            f"Epoch {logs['epoch']}/{self.config['training']['epochs']} | "
            f"{logs['epoch_time_sec']:.1f}s | "
            f"Train Loss: {logs.get('train_loss', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | "
            f"Best: {self.best_metric_value:.4f} (ep {self.best_epoch})"
        )

    # ==================================================================
    # Shared infrastructure
    # ==================================================================

    def _get_augmentation_config(self) -> Dict:
        """Extract spec-augmentation configuration."""
        aug = self.config.get("augmentation", {})
        return {
            "enabled": aug.get("enabled", True),
            "prob": aug.get("prob", 0.5),
            "num_freq_masks": aug.get("num_freq_masks", 2),
            "freq_mask_param": aug.get("freq_mask_param", 6),
            "num_time_masks": aug.get("num_time_masks", 2),
            "time_mask_param": aug.get("time_mask_param", 10),
        }

    def _setup_optimizer(self):
        """Create AdamW from config."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get(
                "weight_decay", self._weight_decay_default
            ),
        )

    def _setup_scheduler(self):
        """Create LR scheduler from config."""
        epochs = self.config["training"]["epochs"]
        scheduler_type = self.config["training"].get("scheduler_type", "cosine")
        warmup_steps = self.config["training"].get("warmup_steps", 0)
        total_steps = max(len(self.train_loader) * epochs, warmup_steps * 2)

        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config["training"].get("min_lr", 1e-6),
        )
        self._scheduler_step_freq = get_scheduler_step_frequency(scheduler_type)

    def _resume_checkpoint(self) -> int:
        """Resume from ``checkpoint_last.pth`` if present.  Returns the
        start epoch (0 when no checkpoint is found)."""
        path = self.run_dir / "checkpoint_last.pth"
        if not path.exists():
            return 0

        print(f"    ↻ Resuming from {path.name}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("precision_state_dict"):
            self.precision.load_state_dict(ckpt["precision_state_dict"])
        if "callbacks_state_dict" in ckpt:
            self.cb_runner.load_state_dict(ckpt["callbacks_state_dict"])

        # ── RNG states for exact reproducibility ────────────────────────
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        if ckpt.get("python_rng_state") is not None:
            random.setstate(ckpt["python_rng_state"])

        # ── Trainer state ───────────────────────────────────────────────
        if "trainer_state" in ckpt:
            self.trainer_state.update(ckpt["trainer_state"])
            self.best_metric_value = self.trainer_state.get(
                "best_metric", self.best_metric_value
            )
            self.best_epoch = self.trainer_state.get("best_epoch", self.best_epoch)
            # Subclass hook for legacy key migration (best_val_acc → best_metric, etc.)
            self._restore_best_from_state(self.trainer_state)

        resume_epoch = ckpt["epoch"]
        print(f"    ✓ Resumed from epoch {resume_epoch + 1}")
        return resume_epoch

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Shared backward → clip → step → update.  Returns grad norm."""
        self.precision.scale_loss(loss).backward()
        self.precision.unscale_gradients(self.optimizer)

        grad_clip = self.config["training"].get("gradient_clip")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=grad_clip if grad_clip is not None else float("inf"),
        ).item()

        self.precision.step(self.optimizer)
        self.precision.update()

        if self.scheduler and self._scheduler_step_freq == "batch":
            self.scheduler.step()

        return grad_norm

    def _update_best(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """Update best-metric tracking.  Writes both the generic
        ``best_metric`` key and the monitor-specific key (e.g.
        ``best_val_acc``) into ``trainer_state`` for backward
        compatibility with callbacks and external tools.  Returns
        True if the metric improved."""
        current = val_metrics.get(self.best_monitor)
        if current is None:
            return False

        improved = (self.best_mode == "min" and current < self.best_metric_value) or (
            self.best_mode == "max" and current > self.best_metric_value
        )
        if improved:
            self.best_metric_value = current
            self.best_epoch = epoch + 1
            # Generic key
            self.trainer_state["best_metric"] = self.best_metric_value
            self.trainer_state["best_epoch"] = self.best_epoch
            # Monitor-specific key for backward compatibility
            self.trainer_state[self.best_monitor] = self.best_metric_value
        return improved

    def _step_scheduler(self, val_metrics: Dict[str, float]):
        """Epoch-level scheduler step (if configured)."""
        if not self.scheduler or self._scheduler_step_freq != "epoch":
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics.get("val_loss", 0.0))
        else:
            self.scheduler.step()

    def _build_logs(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_duration: float,
    ) -> Dict[str, Any]:
        """Merge subclass metrics with common metadata.

        Automatically computes ``samples_per_sec`` when
        ``train_metrics`` includes ``train_total``."""
        logs: Dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_duration,
            "best_metric": self.best_metric_value,
            "best_epoch": self.best_epoch,
        }
        logs.update(train_metrics)
        logs.update(val_metrics)

        # Auto-compute throughput when subclass provides total sample count
        train_total = train_metrics.get("train_total", 0)
        if train_total > 0:
            logs["samples_per_sec"] = train_total / max(epoch_duration, 1e-9)

        logs.update(get_gpu_memory_info(self.device))
        return logs
