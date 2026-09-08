# src/trainers/supervised_transformer_train.py

import json
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import SupervisedBirdSongDataset
from src.evaluation.metrics_collector import MetricsCollector
from src.models import SupervisedTransformer
from .base_trainer import BaseTrainer
from .callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    JSONLoggerCallback,
    CSVLoggerCallback,
    WandBLoggerCallback,
    PlotMetricsCallback,
    _unwrap_compile,
)

# ── helpers ─────────────────────────────────────────────────────────────


def supervised_val_collate_fn(batch):
    """Collate for validation/test batches of (x, y, recording_id)."""
    mel_segments, labels, recording_ids = zip(*batch)
    mel_segments = torch.stack(mel_segments, dim=0)
    labels = torch.stack(labels, dim=0)
    return mel_segments, labels, list(recording_ids)


def aggregate_recordings(logits_all, labels_all, recording_ids_all):
    """
    Aggregate window-level outputs to recording-level predictions.
    O(n) via grouping; avoids repeated full-array scans.
    """
    groups: Dict[str, list] = defaultdict(list)
    for i, rec_id in enumerate(recording_ids_all):
        groups[str(rec_id)].append(i)

    rec_ids, rec_targets, rec_logits = [], [], []
    for rec_id, indices in groups.items():
        rec_ids.append(rec_id)
        rec_targets.append(labels_all[indices[0]].item())
        rec_logits.append(logits_all[indices].mean(dim=0))

    rec_targets = torch.tensor(rec_targets, dtype=torch.long)
    rec_logits = torch.stack(rec_logits, dim=0)
    return rec_ids, rec_targets, rec_logits


# ── trainer ─────────────────────────────────────────────────────────────


class SupervisedTransformerExperimentTrainer(BaseTrainer):
    """Supervised classification training engine with callback-driven
    architecture and recording-level evaluation."""

    _weight_decay_default: float = 0.01

    def __init__(self, config, run_dir, callbacks=None):
        self.best_monitor = "val_acc"
        self.best_mode = "max"
        self.criterion = nn.CrossEntropyLoss()
        super().__init__(config, run_dir, callbacks)

    # ── callbacks ───────────────────────────────────────────────────────
    def _get_default_callbacks(self) -> List[Callback]:
        return [
            EarlyStoppingCallback(
                monitor="val_acc",
                mode="max",
                patience=self.config["training"].get("patience", 15),
            ),
            JSONLoggerCallback(self.run_dir),
            CSVLoggerCallback(self.run_dir),
            PlotMetricsCallback(self.run_dir),
            CheckpointCallback(self.run_dir, monitor="val_acc", mode="max"),
            WandBLoggerCallback(self.config, self.run_dir),
        ]

    # ── legacy checkpoint compatibility ─────────────────────────────────
    def _restore_best_from_state(self, state: Dict):
        if "best_val_acc" in state and "best_metric" not in state:
            self.best_metric_value = state["best_val_acc"]
            state["best_metric"] = self.best_metric_value

    # ── data ────────────────────────────────────────────────────────────
    def get_dataloaders(self, df: pd.DataFrame):
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]
        segment_size = self.config["audio"]["segment_size"]
        window_config = self.config["window"]

        seed = self.config.get("experiment", {}).get("seed", 42)
        test_size = self.config["training"].get("test_size", 0.15)
        val_size = self.config["training"].get("val_size", 0.15)

        min_eval_split = min(val_size, test_size)
        if min_eval_split <= 0:
            raise ValueError("val_size and test_size must be > 0")

        MIN_RECORDINGS_FOR_EVAL = int(np.ceil(1.0 / min_eval_split)) + 1

        counts = df["scientific_name_id"].value_counts()
        eval_classes = counts[counts >= MIN_RECORDINGS_FOR_EVAL].index
        rare_classes = counts[counts < MIN_RECORDINGS_FOR_EVAL].index

        eval_df = df[df["scientific_name_id"].isin(eval_classes)].copy()
        rare_df = df[df["scientific_name_id"].isin(rare_classes)].copy()

        if len(eval_df) == 0:
            print(
                f"WARNING: All {len(rare_classes)} classes have fewer than "
                f"{MIN_RECORDINGS_FOR_EVAL} recordings. "
                f"Training without validation/test split."
            )
            train_df = rare_df.copy()
            val_df = rare_df.iloc[:0].copy()
            test_df = rare_df.iloc[:0].copy()
        else:
            train_df, temp_df = train_test_split(
                eval_df,
                test_size=val_size + test_size,
                random_state=seed,
                stratify=eval_df["scientific_name_id"],
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_size / (val_size + test_size),
                random_state=seed,
                stratify=temp_df["scientific_name_id"],
            )
            train_df = pd.concat([train_df, rare_df], ignore_index=True)

        # ── datasets ────────────────────────────────────────────────────
        train_dataset = SupervisedBirdSongDataset(
            df=train_df,
            segment_size=segment_size,
            train=True,
            spec_aug_config=self._get_augmentation_config(),
            min_db=self.config["audio"]["min_db"],
            max_db=self.config["audio"]["max_db"],
            window_config=window_config,
        )

        self.label_to_idx = train_dataset.label_to_idx
        self.idx_to_label = train_dataset.idx_to_label
        self.class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        val_dataset = None
        if len(val_df) > 0:
            val_dataset = SupervisedBirdSongDataset(
                df=val_df,
                segment_size=segment_size,
                train=False,
                label_to_idx=self.label_to_idx,
                min_db=self.config["audio"]["min_db"],
                max_db=self.config["audio"]["max_db"],
                window_config={
                    "strategy": "sliding",
                    "stride": segment_size,
                },
                return_recording_id=True,
            )

        test_dataset = None
        if len(test_df) > 0:
            test_dataset = SupervisedBirdSongDataset(
                df=test_df,
                segment_size=segment_size,
                train=False,
                label_to_idx=self.label_to_idx,
                min_db=self.config["audio"]["min_db"],
                max_db=self.config["audio"]["max_db"],
                window_config={
                    "strategy": "sliding",
                    "stride": segment_size,
                },
                return_recording_id=True,
            )

        # ── loaders ─────────────────────────────────────────────────────
        loader_kw = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=num_workers > 0,
        )

        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)

        self.val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=supervised_val_collate_fn,
                **loader_kw,
            )

        self.test_loader = None
        if test_dataset is not None and len(test_dataset) > 0:
            self.test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                collate_fn=supervised_val_collate_fn,
                **loader_kw,
            )

    # ── model ───────────────────────────────────────────────────────────
    def _build_model(self):
        model = SupervisedTransformer(
            config=self.config,
            device=str(self.device),
            num_classes=len(self.class_names),
        )
        return model.to(self.device)

    def _on_pre_train(self):
        # Save label mappings for inference / export
        label_map_path = self.run_dir / "label_mappings.json"
        label_map_path.write_text(
            json.dumps(
                {
                    "label_to_idx": self.label_to_idx,
                    "idx_to_label": self.idx_to_label,
                },
                indent=2,
            )
        )

        # Compile after checkpoint weights are loaded
        if self.config["training"].get("compile_model", False):
            self.model = torch.compile(self.model)

    # ── printing ────────────────────────────────────────────────────────
    def _print_train_init(self):
        super()._print_train_init()
        compiled = self.config["training"].get("compile_model", False)
        print(f"    Compiled:  {compiled}")
        print(f"    Classes:   {len(self.class_names)}")
        print(
            f"    Test samples:  "
            f"{len(self.test_loader.dataset) if self.test_loader else 0}"
        )

    def _print_epoch_summary(self, logs):
        epochs = self.config["training"]["epochs"]
        print(
            f"Epoch {logs['epoch']}/{epochs} | "
            f"{logs['epoch_time_sec']:.1f}s | "
            f"Train Loss: {logs['train_loss']:.4f} | "
            f"Train Acc: {logs['train_acc']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val Acc: {logs['val_acc']:.4f} | "
            f"Best: {self.best_metric_value:.4f} (ep {self.best_epoch})"
        )

    # ── training ────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epochs = self.config["training"]["epochs"]
        loss_total = 0.0
        correct = 0
        total = 0
        grad_sum = 0.0
        n_batches = 0

        for batch_idx, (mel_segments, labels) in enumerate(
            tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                leave=False,
            )
        ):
            self.cb_runner.on_batch_begin(self, batch_idx, {"batch": batch_idx})

            mel_segments = mel_segments.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self.precision.autocast():
                logits = self.model(mel_segments)
                loss = self.criterion(logits, labels)

            grad_norm = self._backward_and_step(loss)
            grad_sum += grad_norm
            n_batches += 1

            bs = labels.size(0)
            loss_total += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += bs

            self.cb_runner.on_batch_end(
                self,
                batch_idx,
                {
                    "loss": loss.item(),
                    "grad_norm": grad_norm,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
            )

        return {
            "train_loss": loss_total / max(total, 1),
            "train_acc": correct / max(total, 1),
            "avg_grad_norm": grad_sum / max(n_batches, 1),
            "train_total": total,
        }

    # ── validation ──────────────────────────────────────────────────────
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader.dataset) == 0:
            return {"val_loss": 0.0, "val_acc": 0.0, "val_window_acc": 0.0}

        self.model.eval()
        epochs = self.config["training"]["epochs"]

        val_loss_total = 0.0
        correct_window = 0
        total_windows = 0
        all_logits, all_labels, all_rec_ids = [], [], []

        with torch.no_grad():
            for mel_segments, labels, recording_ids in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False,
            ):
                mel_segments = mel_segments.to(self.device)
                labels = labels.to(self.device)

                with self.precision.autocast():
                    logits = self.model(mel_segments)

                loss = self.criterion(logits, labels)
                val_loss_total += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_window += (preds == labels).sum().item()
                total_windows += labels.size(0)

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.cpu())
                all_rec_ids.extend(recording_ids)

        avg_val_loss = 0.0
        val_acc = 0.0
        val_window_acc = 0.0

        if total_windows > 0:
            all_logits_t = torch.cat(all_logits, dim=0)
            all_labels_t = torch.cat(all_labels, dim=0)

            rec_ids, rec_targets, rec_logits = aggregate_recordings(
                all_logits_t, all_labels_t, all_rec_ids
            )
            rec_loss = self.criterion(
                rec_logits.to(self.device), rec_targets.to(self.device)
            )
            avg_val_loss = rec_loss.item()
            rec_preds = rec_logits.argmax(dim=1).cpu()
            val_acc = (rec_preds == rec_targets).float().mean().item()
            val_window_acc = correct_window / total_windows

        return {
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_window_acc": val_window_acc,
        }

    # ── post-train (test evaluation) ────────────────────────────────────
    def _post_train(self) -> Dict[str, Any]:
        if self.test_loader is None or len(self.test_loader.dataset) == 0:
            print("WARNING: Test set is empty. Skipping final evaluation.")
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "warning": "empty_test_set",
            }

        best_ckpt_path = self.run_dir / "checkpoint_best.pth"
        if not best_ckpt_path.exists():
            print("WARNING: No best checkpoint found. Using current model weights.")
        else:
            best_ckpt = torch.load(best_ckpt_path, weights_only=False)
            _unwrap_compile(self.model).load_state_dict(best_ckpt["model_state_dict"])

        return self._evaluate(self.model, self.test_loader, self.class_names)

    # ── evaluation ──────────────────────────────────────────────────────
    def _evaluate(
        self, model: nn.Module, test_loader: DataLoader, class_names: list
    ) -> Dict:
        model.eval()
        collector = MetricsCollector(self.run_dir, class_names)

        all_logits, all_labels, all_rec_ids = [], [], []

        with torch.no_grad():
            for batch_idx, (mel_segments, labels, recording_ids) in enumerate(
                tqdm(test_loader, desc="Evaluating", leave=False)
            ):
                mel_segments = mel_segments.to(self.device)
                with self.precision.autocast():
                    logits = model(mel_segments)

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.cpu())
                all_rec_ids.extend(recording_ids)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        rec_ids, rec_targets, rec_logits = aggregate_recordings(
            all_logits, all_labels, all_rec_ids
        )

        probs = torch.softmax(rec_logits, dim=1).numpy()
        preds = rec_logits.argmax(dim=1).numpy()
        targets = rec_targets.numpy()

        collector.add_batch(preds, targets, probs)
        metrics = collector.compute_metrics()
        collector.save_metrics_json()
        collector.plot_confusion_matrix()
        collector.plot_per_class_metrics()
        collector.generate_markdown_report()
        return metrics
