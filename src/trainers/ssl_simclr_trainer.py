# src/trainers/ssl_simclr_train.py

from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.datasets.ssl import SimCLRDataset, simclr_collate_fn
from src.models.ssl.simclr import SimCLR
from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead
from .base_trainer import BaseTrainer
from .callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    JSONLoggerCallback,
    CSVLoggerCallback,
    WandBLoggerCallback,
    PlotMetricsCallback,
)


class SimCLRExperimentTrainer(BaseTrainer):
    """
    Trainer for self-supervised contrastive learning with SimCLR.
    Trains an encoder + projection head; ``best_model.pth`` contains
    only the encoder weights for downstream use.
    """

    def __init__(self, config, run_dir, callbacks=None):
        self.best_monitor = "val_loss"
        self.best_mode = "min"
        super().__init__(config, run_dir, callbacks)

    # ── callbacks ───────────────────────────────────────────────────────
    def _get_default_callbacks(self) -> List[Callback]:
        return [
            EarlyStoppingCallback(
                monitor="val_loss",
                mode="min",
                patience=self.config["training"].get("patience", 15),
            ),
            JSONLoggerCallback(self.run_dir),
            CSVLoggerCallback(self.run_dir),
            PlotMetricsCallback(self.run_dir),
            CheckpointCallback(
                run_dir=self.run_dir,
                monitor="val_loss",
                mode="min",
                # Strip projection head — only the encoder is useful
                # after pretraining.
                state_dict_fn=lambda m: m.encoder.state_dict(),
            ),
            WandBLoggerCallback(self.config, self.run_dir),
        ]

    # ── legacy checkpoint compatibility ─────────────────────────────────
    def _restore_best_from_state(self, state: Dict):
        if "best_loss" in state and "best_metric" not in state:
            self.best_metric_value = state["best_loss"]
            state["best_metric"] = self.best_metric_value

    # ── data ────────────────────────────────────────────────────────────
    def get_dataloaders(self, df):
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]
        segment_size = self.config["audio"]["segment_size"]
        window_config = self.config.get("window", {})
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

        loader_kw = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
            collate_fn=simclr_collate_fn,
        )
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

    # ── model ───────────────────────────────────────────────────────────
    def _build_model(self):
        enc_cfg = self.config["model"]
        proj_cfg = self.config.get("projection", {})

        encoder = CNNEncoder(
            n_mels=self.config["audio"]["n_mels"],
            embed_dim=enc_cfg.get("embed_dim", 512),
            base_channels=enc_cfg.get("base_channels", 64),
            dropout=enc_cfg.get("dropout", 0.1),
        )
        projection = ProjectionHead(
            input_dim=encoder.get_output_dim(),
            hidden_dim=proj_cfg.get("hidden_dim", 256),
            output_dim=proj_cfg.get("output_dim", 128),
        )
        temperature = self.config["training"].get("temperature", 0.07)

        model = SimCLR(
            encoder=encoder, projection=projection, temperature=temperature
        )
        return model.to(self.device)

    # ── printing ────────────────────────────────────────────────────────
    def _print_train_init(self):
        super()._print_train_init()
        print(f"    Temperature: {self.model.temperature}")

    def _print_epoch_summary(self, logs):
        epochs = self.config["training"]["epochs"]
        print(
            f"Epoch {logs['epoch']}/{epochs} | "
            f"{logs['epoch_time_sec']:.1f}s | "
            f"Train Loss: {logs['train_loss']:.4f} | "
            f"Train Contrastive Acc: {logs['train_contrastive_acc']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val Contrastive Acc: {logs['val_contrastive_acc']:.4f} | "
            f"Best: {self.best_metric_value:.4f} (ep {self.best_epoch})"
        )

    # ── training ────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epochs = self.config["training"]["epochs"]
        loss_total = 0.0
        acc_total = 0.0
        grad_sum = 0.0
        n_batches = 0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False,
        )
        for x1, x2 in pbar:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self.precision.autocast():
                loss, acc = self.model.training_step(x1, x2)

            grad_norm = self._backward_and_step(loss)
            grad_sum += grad_norm

            bs = x1.size(0)
            loss_total += loss.item() * bs
            acc_total += acc.item() * bs
            total_samples += bs
            n_batches += 1

            pbar.set_postfix(loss=loss.item(), acc=acc.item())

        n = max(len(self.train_loader.dataset), 1)
        return {
            "train_loss": loss_total / n,
            "train_contrastive_acc": acc_total / n,
            "avg_grad_norm": grad_sum / max(n_batches, 1),
            "train_total": total_samples,
        }

    # ── validation ──────────────────────────────────────────────────────
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        epochs = self.config["training"]["epochs"]
        loss_total = 0.0
        acc_total = 0.0

        with torch.no_grad():
            for x1, x2 in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False,
            ):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                with self.precision.autocast():
                    loss, acc = self.model.training_step(x1, x2)
                loss_total += loss.item() * x1.size(0)
                acc_total += acc.item() * x1.size(0)

        n = max(len(self.val_loader.dataset), 1)
        return {
            "val_loss": loss_total / n,
            "val_contrastive_acc": acc_total / n,
        }

    # ── post-train ──────────────────────────────────────────────────────
    def _post_train(self) -> Dict[str, Any]:
        return {
            "val_loss": self._last_val_metrics.get("val_loss", 0),
            "val_contrastive_acc": self._last_val_metrics.get(
                "val_contrastive_acc", 0
            ),
            "train_loss": self._last_train_metrics.get("train_loss", 0),
            "train_contrastive_acc": self._last_train_metrics.get(
                "train_contrastive_acc", 0
            ),
            "best_loss": self.best_metric_value,
            "best_epoch": self.best_epoch,
        }
