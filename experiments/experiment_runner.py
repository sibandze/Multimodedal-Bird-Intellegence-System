# experiments/experiment_runner.py
"""Orchestrate multiple training experiments with different configurations."""

import os
import sys
import json
import traceback
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import random
import warnings

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List
import csv
from tqdm import tqdm

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.configs import (
    load_and_resolve_config,
    resolve_metadata_csv_path,
    set_nested_config,
)
from experiments.sweep_configs import SWEEP_SUITES
from src.trainers import SupervisedTransformerExperimentTrainer
from src.trainers import SimCLRExperimentTrainer

# Which suites are SSL vs supervised (by naming convention)
SSL_SUITE_PREFIXES = ("ssl_",)


def is_ssl_suite(sweep_name: str) -> bool:
    """Check if a sweep suite name indicates SSL training."""
    return any(sweep_name.startswith(prefix) for prefix in SSL_SUITE_PREFIXES)


class ExperimentManager:
    """Manages experiment runs and result collection."""

    ROOT_DIR = Path(__file__).parent.parent

    def __init__(
        self,
        base_config_path: str,
        results_dir: str = "results",
        mode: str = "auto",
    ):
        """
        Args:
            mode: 'supervised', 'ssl', or 'auto'.
                  'auto' infers from suite name prefix.
        """
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.mode = mode

        # Initialize seed attribute before set_seed() or load_data() are called
        self.seed = 42

        # Load base config
        config_rel_path = (
            str(self.base_config_path.relative_to(self.ROOT_DIR))
            if self.base_config_path.is_absolute()
            else str(self.base_config_path)
        )
        self.base_config = load_and_resolve_config(self.ROOT_DIR, config_rel_path)

        self.experiment_name = None
        self.experiment_dir = None
        self.results_csv = None
        self.run_counter = 0

        self.df = None

    def set_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self):
        """Load and distill dataset to num_classes and num_samples_per_class."""
        if self.df is None:
            csv_path = resolve_metadata_csv_path(self.base_config)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Processed CSV not found at {csv_path}.",
                    f"Run data pipeline first: python main.py --pipeline",
                )
            full_df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(full_df)} samples from {csv_path}")

            # ---- Distillation ----
            data_cfg = self.base_config.get("data", {})
            requested_classes = data_cfg.get("num_classes")
            requested_samples = data_cfg.get("num_samples_per_class")
            label_col = data_cfg.get("label_column")

            if requested_classes and requested_samples:
                print(
                    f"🔬 Distilling dataset to {requested_classes} classes x {requested_samples} samples each"
                )

                # Group by label
                grouped = full_df.groupby(label_col)
                class_counts = grouped.size().sort_values(ascending=False)

                # Check how many classes have enough samples
                eligible_classes = class_counts[class_counts >= requested_samples]

                if len(eligible_classes) < requested_classes:
                    # Auto-reduce to the highest common number
                    # Find the maximum samples per class available for the top requested_classes
                    top_class_counts = class_counts.head(requested_classes)
                    max_possible = int(top_class_counts.min())

                    warnings.warn(
                        f"Only {len(eligible_classes)} classes have at least {requested_samples} samples. "
                        f"Auto-reducing samples per class to {max_possible} (max common across top {requested_classes} classes)."
                    )
                    requested_samples = max_possible

                    # Re-check eligibility with reduced samples
                    eligible_classes = class_counts[class_counts >= requested_samples]

                    if len(eligible_classes) < requested_classes:
                        # If still not enough, reduce number of classes too
                        new_num_classes = len(eligible_classes)
                        warnings.warn(
                            f"Still only {len(eligible_classes)} classes available with {requested_samples} samples. "
                            f"Auto-reducing num_classes to {new_num_classes}."
                        )
                        requested_classes = new_num_classes

                selected_classes = eligible_classes.index[:requested_classes].tolist()
                print(
                    f"   Selected {len(selected_classes)} classes with {requested_samples} samples each"
                )
                print(f"   Classes: {selected_classes}")

                # Sample per class
                sampled_dfs = []
                rng = np.random.RandomState(self.seed)
                for cls in selected_classes:
                    cls_df = full_df[full_df[label_col] == cls]
                    sampled = cls_df.sample(n=requested_samples, random_state=rng)
                    sampled_dfs.append(sampled)

                self.df = pd.concat(sampled_dfs, ignore_index=True)
                print(f"✓ Final distilled dataset: {len(self.df)} samples")

                # Optional: shuffle the final dataset
                self.df = self.df.sample(frac=1, random_state=rng).reset_index(
                    drop=True
                )

                # Update config with actual values used
                self.base_config["data"]["num_classes"] = requested_classes
                self.base_config["data"]["num_samples_per_class"] = requested_samples
            else:
                # If no distillation requested, use full dataset
                self.df = full_df

        return self.df

    def _resolve_mode(self, sweep_name: str) -> str:
        """Determine training mode for a given sweep."""
        if self.mode == "auto":
            return "ssl" if is_ssl_suite(sweep_name) else "supervised"
        return self.mode

    def create_experiment_run(
        self,
        sweep_name: str,
        run_index: int,
        hyperparams: Dict[str, Any],
        run_seed: int,
    ) -> tuple:
        """Create a unique directory and config for this experiment run."""
        if not self.experiment_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"exp_{timestamp}"
            self.experiment_dir = self.results_dir / self.experiment_name
            self.experiment_dir.mkdir(exist_ok=True, parents=True)
            self.results_csv = self.experiment_dir / "results.csv"

        run_name = f"run_{run_index:04d}_{sweep_name}"
        run_dir = self.experiment_dir / run_name
        run_dir.mkdir(exist_ok=True, parents=True)

        run_config = self._merge_config_with_hyperparams(
            hyperparams, run_dir, run_name, run_seed
        )

        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)

        return run_dir, run_config

    def _merge_config_with_hyperparams(
        self,
        hyperparams: Dict[str, Any],
        run_dir: Path,
        run_name: str,
        run_seed: int,
    ) -> Dict:
        """
        Merge base config with hyperparameter overrides.

        Sweep params use dotted paths (e.g. "training.learning_rate") which
        are applied directly via set_nested_config(). No mapping dict needed.
        """
        config = yaml.safe_load(yaml.dump(self.base_config))  # Deep copy

        # Ensure sections exist
        for section in (
            "training",
            "model",
            "augmentation",
            "logging",
            "projection",
            "experiment",
        ):
            if section not in config:
                config[section] = {}

        # Apply sweep hyperparameters via dotted-path keys
        for dotted_path, param_value in hyperparams.items():
            set_nested_config(config, dotted_path, param_value)

        # Per-run seed for reproducible, distinct runs
        config["experiment"]["seed"] = run_seed

        # Configure logging
        config["logging"]["wandb_run_name"] = run_name
        config["logging"]["wandb_run_id"] = f"{self.experiment_name}_{run_name}"
        if "wandb_project" not in config["logging"]:
            config["logging"]["wandb_project"] = "bird-song-classifier"

        # Experiment metadata
        config["experiment"]["name"] = run_name
        config["experiment"]["experiment_group"] = self.experiment_name
        config["experiment"]["run_dir"] = str(run_dir)
        config["experiment"]["timestamp"] = datetime.now().isoformat()
        config["experiment"]["hyperparams"] = hyperparams

        return config

    def log_run_result(
        self, run_index: int, sweep_name: str, hyperparams: Dict, metrics: Dict
    ):
        """Append run results to the results CSV."""
        row = {
            "run_id": run_index,
            "sweep_name": sweep_name,
            "timestamp": datetime.now().isoformat(),
        }
        row.update(hyperparams)
        row.update(metrics)

        if not self.results_csv.exists():
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
        else:
            # Handle case where new columns appear in later runs
            existing_rows = []
            with open(self.results_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                for r in reader:
                    existing_rows.append(r)

            all_fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))

            with open(self.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
                writer.writerow(row)

        self.run_counter += 1

    def run_experiment(
        self,
        run_index: int,
        sweep_name: str,
        hyperparams: Dict,
        dry_run: bool = False,
    ):
        """Run a single training experiment."""
        # FIX: Per-run seed so each experiment has deterministic but distinct randomness
        run_seed = self.seed + run_index
        self.set_seed(run_seed)

        run_dir, run_config = self.create_experiment_run(
            sweep_name, run_index, hyperparams, run_seed
        )

        mode = self._resolve_mode(sweep_name)

        if dry_run:
            print(f"  [{run_index}] [DRY RUN] [{mode}] Would train with: {hyperparams}")
            print(f"      Seed: {run_seed}")
            print(f"      Config: {run_dir}/config.yaml")
            return None

        try:
            if mode == "ssl":
                trainer = SimCLRExperimentTrainer(run_config, run_dir)
            else:
                trainer = SupervisedTransformerExperimentTrainer(run_config, run_dir)

            print(f"\n  [{run_index}] [{mode}] Training: {hyperparams}")
            print(f"      Seed: {run_seed}")
            metrics = trainer.train(self.df)

            self.log_run_result(run_index, sweep_name, hyperparams, metrics)

            # Print mode-appropriate summary
            if mode == "ssl":
                print(
                    f"      ✓ Val Loss: {metrics.get('val_loss', 0):.4f} | "
                    f"Contrastive Acc: {metrics.get('val_contrastive_acc', 0):.4f}"
                )
            else:
                print(
                    f"      ✓ Accuracy: {metrics.get('accuracy', 0):.4f} | "
                    f"Macro F1: {metrics.get('macro_f1', 0):.4f}"
                )

            return metrics

        except Exception as e:
            tb = traceback.format_exc()
            print(f"      ✗ Error during training: {str(e)}\n{tb}")

            if mode == "ssl":
                error_metrics = {
                    "val_loss": float("inf"),
                    "val_contrastive_acc": 0.0,
                    "error": str(e),
                    "error_traceback": tb,
                }
            else:
                error_metrics = {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "weighted_f1": 0.0,
                    "error": str(e),
                    "error_traceback": tb,
                }

            self.log_run_result(run_index, sweep_name, hyperparams, error_metrics)
            return None

    def save_experiment_summary(self, mode: str = "supervised"):
        """Generate a summary of the experiment."""
        summary_path = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        summary = (
            f"# Experiment Summary\n"
            f"\n"
            f"**Experiment ID:** {self.experiment_name}\n"
            f"**Mode:** {mode}\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Total Runs:** {self.run_counter}\n"
            f"**Base Seed:** {self.seed}\n"
            f"\n"
            f"## Results Location\n"
            f"- Detailed results: `{self.results_csv}`\n"
            f"- Run directories: `{self.experiment_dir}/run_XXXX_*/`\n"
            f"\n"
        )

        if self.results_csv.exists():
            try:
                df_results = pd.read_csv(self.results_csv)

                if mode == "ssl":
                    sort_col = "val_loss"
                    ascending = True
                    summary += "### Top 5 Runs by Val Loss (lowest first)\n\n"
                    summary += "| Run ID | Val Loss | Contrastive Acc | LR | Batch Size | Seed |\n"
                    summary += "|--------|----------|-----------------|-----|------------|------|\n"

                    df_sorted = df_results.sort_values(sort_col, ascending=ascending)
                    for _, row in df_sorted.head(5).iterrows():
                        summary += (
                            f"| {int(row['run_id'])} "
                            f"| {row.get('val_loss', 0):.4f} "
                            f"| {row.get('val_contrastive_acc', 0):.4f} "
                            f"| {row.get('training.learning_rate', 'N/A')} "
                            f"| {row.get('training.batch_size', 'N/A')} "
                            f"| {row.get('experiment.seed', 'N/A')} |\n"
                        )
                else:
                    sort_col = "accuracy"
                    ascending = False
                    summary += "### Top 5 Runs by Accuracy\n\n"
                    summary += "| Run ID | Accuracy | Macro F1 | Learning Rate | Batch Size | Seed |\n"
                    summary += "|--------|----------|----------|---------------|------------|------|\n"

                    df_sorted = df_results.sort_values(sort_col, ascending=ascending)
                    for _, row in df_sorted.head(5).iterrows():
                        summary += (
                            f"| {int(row['run_id'])} "
                            f"| {row.get('accuracy', 0):.4f} "
                            f"| {row.get('macro_f1', 0):.4f} "
                            f"| {row.get('training.learning_rate', 'N/A')} "
                            f"| {row.get('training.batch_size', 'N/A')} "
                            f"| {row.get('experiment.seed', 'N/A')} |\n"
                        )

            except Exception as e:
                summary += f"(Error reading results: {e})\n"

        summary += (
            f"\n"
            f"## Instructions\n"
            f"\n"
            f"1. Review `results.csv` for aggregate metrics across all runs\n"
            f"2. Inspect individual `run_XXXX_*/` directories for:\n"
            f"   - `config.yaml` - exact hyperparameters used\n"
            f"   - `label_mappings.json` - class name to index mapping\n"
            f"   - `checkpoint_best.pth` - trained model checkpoint\n"
            f"   - `training_metrics.json` - epoch-by-epoch training logs\n"
        )

        if mode == "ssl":
            summary += (
                f"   - `checkpoint_best.pth` - encoder-only weights (projection head excluded)\n"
                f"\n"
                f"## Next Steps\n"
                f"\n"
                f"After reviewing SSL results:\n"
                f"- Use best encoder checkpoint for linear probing evaluation\n"
                f"- Compare against supervised baseline\n"
            )
        else:
            summary += (
                f"   - `evaluation_metrics.json` - final test metrics\n"
                f"   - `confusion_matrix.png` - confusion matrix visualization\n"
                f"\n"
                f"## Next Steps\n"
                f"\n"
                f"After reviewing results:\n"
                f"- Identify best-performing configurations\n"
                f"- Use those as baseline for contrastive learning experiments\n"
            )

        with open(summary_path, "w") as f:
            f.write(summary)

        print(f"\n✓ Experiment summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep experiments")
    parser.add_argument(
        "--suite",
        type=str,
        default="quick_baseline",
        choices=list(SWEEP_SUITES.keys()),
        help="Which sweep suite to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["supervised", "ssl", "auto"],
        help="Training mode: supervised, ssl, or auto (inferred from suite name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Initialize experiment manager
    manager = ExperimentManager(args.config, args.results_dir, mode=args.mode)
    manager.set_seed(args.seed)

    # Get sweep suite
    sweep_suite = SWEEP_SUITES[args.suite]

    # Resolve effective mode for display
    effective_mode = manager._resolve_mode(args.suite)

    print(f"\n{'='*80}")
    print(f"🚀 Running Experiment Suite: {args.suite}")
    print(f"   Mode: {effective_mode}")
    print(f"   Base Seed: {args.seed}")
    print(f"{'='*80}\n")

    total_runs = sum(len(sweep.generate_configs()) for sweep in sweep_suite)
    print(f"📊 Total configurations to run: {total_runs}")

    if not args.dry_run:
        print(
            f"⚠️  This will take approximately {total_runs * 10} minutes "
            f"(assuming ~10 min/run)"
        )
        print(f"💾 Results will be saved to: {manager.results_dir}\n")

        try:
            manager.load_data()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    run_index = 1

    for sweep in sweep_suite:
        print(f"\n{'─'*80}")
        print(f"📋 Sweep: {sweep.name}")
        print(f"   Description: {sweep.description}")
        print(f"{'─'*80}")

        configs = sweep.generate_configs()
        print(f"   Configurations: {len(configs)}\n")

        for config_idx, hyperparams in enumerate(configs, 1):
            manager.run_experiment(
                run_index, sweep.name, hyperparams, dry_run=args.dry_run
            )
            run_index += 1

    # Save summary with mode context
    manager.save_experiment_summary(mode=effective_mode)

    print(f"\n{'='*80}")
    print(f"✅ Experiment complete!")
    print(f"📁 Results saved to: {manager.experiment_dir}")
    print(f"{'='*80}\n")

    print("📖 Next Steps:")
    print(f"   1. cd results/{manager.experiment_name}")
    print(f"   2. cat results.csv | head -20  # View top results")
    print(f"   3. Review EXPERIMENT_SUMMARY.md for overview")
    print(f"   4. Inspect individual run_XXXX_*/ directories for details")
    print()


if __name__ == "__main__":
    main()
