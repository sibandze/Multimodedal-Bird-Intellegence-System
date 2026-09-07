# src/training/ssl_simclr_train.py

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets.ssl import SimCLRDataset, simclr_collate_fn
from src.models.ssl.simclr import SimCLR
from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead
from src.training.precision import PrecisionManager
from src.training.scheduler import create_scheduler, get_scheduler_step_frequency
from src.utils.memory_utils import get_gpu_memory_info, log_memory_usage
from src.training.callbacks import (
    Callback,
    CallbackRunner,
    CheckpointCallback,
    EarlyStoppingCallback,
    JSONLoggerCallback,
    CSVLoggerCallback,
    WandBLoggerCallback,
    PlotMetricsCallback,
)


class SimCLRExperimentTrainer:
    """
    Trainer for self-supervised contrastive learning with SimCLR.
    Trains an encoder + projection head, then evaluates using a linear probe
    on the frozen encoder (optional, can be added later).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(exist_ok=True, parents=True)

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

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.stop_training = False

        if callbacks is None:
            callbacks = [
                EarlyStoppingCallback(
                    monitor="val_loss",
                    mode="min",
                    patience=config["training"].get("patience", 15),
                ),
                JSONLoggerCallback(self.run_dir),
                CSVLoggerCallback(self.run_dir),
                PlotMetricsCallback(self.run_dir),
                CheckpointCallback(
                    run_dir=self.run_dir,
                    monitor="val_loss",
                    mode="min",
                    # Strip projection head from best_model.pth — only the
                    # encoder is useful after pretraining.
                    state_dict_fn=lambda m: m.encoder.state_dict(),
                ),
                WandBLoggerCallback(config, self.run_dir),
            ]
        self.cb_runner = CallbackRunner(callbacks)

    def request_stop(self):
        self.stop_training = True

    def get_dataloaders(self, df):
        """Build SSL dataloaders – no labels needed."""
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]
        segment_size = self.config["audio"]["segment_size"]
        window_config = self.config.get("window", {})

        from sklearn.model_selection import train_test_split

        seed = self.config.get("experiment", {}).get("seed", 42)

        train_df, val_df = train_test_split(
            df, test_size=0.05, random_state=seed
        )

        train_dataset = SimCLRDataset(
            df=train_df,
            segment_size=segment_size,
            min_db=self.config["audio"]["min_db"],
            max_db=self.config["audio"]["max_db"],
            train=True,
            apply_augmentation=True,
            window_config=window_config,
            acoustic_aug_config=self.config.get("acoustic_augmentation", {}),
            spec_aug_config=self._get_augmentation_config(),
        )

        val_dataset = SimCLRDataset(
            df=val_df,
            segment_size=segment_size,
            min_db=self.config["audio"]["min_db"],
            max_db=self.config["audio"]["max_db"],
            train=False,
            apply_augmentation=True,
            window_config={"strategy": "sliding", "stride": segment_size},
            acoustic_aug_config=self.config.get("acoustic_augmentation", {}),
            spec_aug_config=self._get_augmentation_config(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
            collate_fn=simclr_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
            collate_fn=simclr_collate_fn,
        )
        return train_loader, val_loader

    def _get_augmentation_config(self):
        aug_cfg = self.config.get("augmentation", {})
        return {
            "enabled": aug_cfg.get("enabled", True),
            "prob": aug_cfg.get("prob", 0.5),
            "num_freq_masks": aug_cfg.get("num_freq_masks", 2),
            "freq_mask_param": aug_cfg.get("freq_mask_param", 6),
            "num_time_masks": aug_cfg.get("num_time_masks", 2),
            "time_mask_param": aug_cfg.get("time_mask_param", 10),
        }

    def _build_model(self):
        """Build SimCLR model from config."""
        encoder_cfg = self.config["model"]
        proj_cfg = self.config.get("projection", {})

        encoder = CNNEncoder(
            n_mels=self.config["audio"]["n_mels"],
            embed_dim=encoder_cfg.get("embed_dim", 512),
            base_channels=encoder_cfg.get("base_channels", 64),
            dropout=encoder_cfg.get("dropout", 0.1),
        )
        projection = ProjectionHead(
            input_dim=encoder.get_output_dim(),
            hidden_dim=proj_cfg.get("hidden_dim", 256),
            output_dim=proj_cfg.get("output_dim", 128),
        )

        temperature = self.config["training"].get("temperature", 0.07)

        model = SimCLR(
            encoder=encoder,
            projection=projection,
            temperature=temperature,
        )
        return model.to(self.device)

    def train(self, df):
        """Run SimCLR training loop."""
        train_loader, val_loader = self.get_dataloaders(df)

        self.model = self._build_model()

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n>>> Initializing SimCLR Training:")
        print(f"    Device: {self.device}")
        print(f"    Precision: {self.precision.precision_name()}")
        print(f"    Trainable params: {num_params:,}")
        print(f"    Train samples: {len(train_loader.dataset)}")
        print(f"    Val samples: {len(val_loader.dataset)}")
        print(f"    Temperature: {self.model.temperature}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"].get("weight_decay", 1e-4),
        )
        epochs = self.config["training"]["epochs"]
        scheduler_type = self.config["training"].get("scheduler_type", "cosine")
        warmup_steps = self.config["training"].get("warmup_steps", 0)
        total_steps = max(len(train_loader) * epochs, warmup_steps * 2)
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config["training"].get("min_lr", 1e-6),
        )
        step_frequency = get_scheduler_step_frequency(scheduler_type)

        # Resume from checkpoint
        resume_epoch = 0
        checkpoint_path = self.run_dir / "checkpoint_last.pth"
        if checkpoint_path.exists():
            print(f"    ↻ Resuming from {checkpoint_path.name}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("precision_state_dict"):
                self.precision.load_state_dict(checkpoint["precision_state_dict"])
            if "callbacks_state_dict" in checkpoint:
                self.cb_runner.load_state_dict(checkpoint["callbacks_state_dict"])
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            if checkpoint.get("numpy_rng_state") is not None:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if checkpoint.get("python_rng_state") is not None:
                random.setstate(checkpoint["python_rng_state"])

            checkpoint_logs = checkpoint.get("logs", {})

            self.best_loss = checkpoint_logs.get("best_loss", self.best_loss)
            self.best_epoch = checkpoint_logs.get("best_epoch", self.best_epoch)

            resume_epoch = checkpoint["epoch"]
            print(f"    ✓ Resumed successfully from epoch {resume_epoch + 1}")

        self.cb_runner.on_train_begin(self)

        for epoch in range(resume_epoch, epochs):
            if self.stop_training:
                break

            train_loader.dataset.set_epoch(epoch)

            self.cb_runner.on_epoch_begin(self, epoch)
            epoch_start = time.time()

            # --- Training ---
            self.model.train()
            train_loss_total = 0.0
            train_acc_total = 0.0
            num_batches = 0
            grad_norm_sum = 0.0

            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
            )
            for batch_idx, (x1, x2) in enumerate(pbar):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with self.precision.autocast():
                    loss, acc = self.model.training_step(x1, x2)

                self.precision.scale_loss(loss).backward()
                self.precision.unscale_gradients(self.optimizer)

                grad_clip = self.config["training"].get("gradient_clip")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_clip if grad_clip is not None else float("inf"),
                ).item()

                grad_norm_sum += grad_norm

                self.precision.step(self.optimizer)
                self.precision.update()

                if self.scheduler and step_frequency == "batch":
                    self.scheduler.step()

                train_loss_total += loss.item() * x1.size(0)
                train_acc_total += acc.item() * x1.size(0)
                num_batches += 1

                pbar.set_postfix(loss=loss.item(), acc=acc.item())

            avg_train_loss = train_loss_total / max(len(train_loader.dataset), 1)
            avg_train_acc = train_acc_total / max(len(train_loader.dataset), 1)

            # --- Validation ---
            self.model.eval()
            val_loss_total = 0.0
            val_acc_total = 0.0
            with torch.no_grad():
                for x1, x2 in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False
                ):
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    with self.precision.autocast():
                        loss, acc = self.model.training_step(x1, x2)
                    val_loss_total += loss.item() * x1.size(0)
                    val_acc_total += acc.item() * x1.size(0)

            avg_val_loss = val_loss_total / max(len(val_loader.dataset), 1)
            avg_val_acc = val_acc_total / max(len(val_loader.dataset), 1)

            # Update trainer-level best tracking
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_epoch = epoch + 1

            if self.scheduler and step_frequency == "epoch":
                self.scheduler.step(avg_val_loss)

            epoch_duration = time.time() - epoch_start
            logs = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_contrastive_acc": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_contrastive_acc": avg_val_acc,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm_sum / num_batches if num_batches > 0 else 0.0,
                "epoch_time_sec": epoch_duration,
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
            }
            logs.update(get_gpu_memory_info(self.device))

            print(
                f"Epoch {epoch+1}/{epochs} | {epoch_duration:.1f}s | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Contrastive Acc: {avg_train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Contrastive Acc: {avg_val_acc:.4f} | "
                f"Best: {self.best_loss:.4f} (ep {self.best_epoch})"
            )

            # CheckpointCallback handles all saving:
            #   checkpoint_last.pth  (every epoch, full model for resumption)
            #   checkpoint_best.pth  (on improvement, full model for resumption)
            #   best_model.pth       (on improvement, encoder-only for inference)
            self.cb_runner.on_epoch_end(self, epoch, logs)

        self.cb_runner.on_train_end(self)
        return {
            "val_loss": avg_val_loss,
            "val_contrastive_acc": avg_val_acc,
            "train_loss": avg_train_loss,
            "train_contrastive_acc": avg_train_acc,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }
