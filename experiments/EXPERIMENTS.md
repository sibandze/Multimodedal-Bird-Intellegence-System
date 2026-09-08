```markdown
# Experiments & Evaluation Pipeline

This document describes the comprehensive experiments framework for the
Multimodal Bird Intelligence System. The pipeline supports both **supervised
classification** and **self-supervised learning (SimCLR)** experiments with
systematic hyperparameter sweeps.

## Overview

The experiments pipeline enables:

1. **Systematic hyperparameter sweeps** with reproducible, per-run configurations
2. **Dual-mode training** — supervised classification and SSL (SimCLR contrastive)
3. **Comprehensive evaluation metrics** at the per-class and overall level
4. **Detailed logging** of all runs for later comparison
5. **Data distillation** — subsample N classes × M samples for fast iteration
6. **Baseline establishment** for comparing supervised vs. contrastive approaches

## Directory Structure

```
experiments/
├── sweep_configs.py          # Hyperparameter sweep definitions
├── experiment_runner.py       # Main experiment orchestration
├── EXPERIMENTS.md            # This file
└── results/                  # Output directory (created at runtime)
    └── exp_YYYYMMDD_HHMMSS/  # Timestamped experiment batch
        ├── EXPERIMENT_SUMMARY.md
        ├── results.csv              # Aggregate results across all runs
        └── run_0000_sweep_name/    # Individual run directories
            ├── config.yaml          # Exact config used for this run
            ├── best_model.pth       # Trained model checkpoint
            ├── label_mappings.json  # Class-name ↔ index mapping
            ├── training_metrics.json # Epoch-by-epoch logs
            ├── evaluation_metrics.json  # Final test metrics (supervised)
            ├── EVALUATION_REPORT.md     # Markdown evaluation report
            ├── confusion_matrix.png
            └── per_class_metrics.png
```

## Quick Start

### 1. Dry-Run a Sweep

Preview what would run without actually training:

```bash
python -m experiments.experiment_runner \
  --suite quick_baseline \
  --config configs/config.yaml \
  --dry-run
```

### 2. Supervised Baseline

```bash
python -m experiments.experiment_runner \
  --suite standard_baseline \
  --config configs/config.yaml \
  --seed 42
```

### 3. SSL (SimCLR) Sanity Check

Verify the contrastive learning pipeline works end-to-end:

```bash
python -m experiments.experiment_runner \
  --suite ssl_sanity \
  --config configs/config.yaml \
  --mode ssl \
  --seed 42
```

### 4. Comprehensive SSL Sweep

```bash
python -m experiments.experiment_runner \
  --suite ssl_comprehensive \
  --config configs/config.yaml \
  --mode ssl \
  --seed 42
```

## Command-Line Reference

```
python -m experiments.experiment_runner [OPTIONS]

Options:
  --suite       Sweep suite to run (see Available Sweep Suites below)
  --mode        Training mode: supervised | ssl | auto (default: auto)
                'auto' infers from suite name prefix (ssl_* → SSL, else supervised)
  --config      Path to base config file (default: configs/config.yaml)
  --results-dir Directory for experiment outputs (default: results/)
  --dry-run     Print configurations without training
  --seed        Base random seed (default: 42)
```

## Mode Selection

The runner supports two training modes:

| Mode         | Trainer                      | Metrics Tracked                                          |
|--------------|------------------------------|----------------------------------------------------------|
| `supervised` | `SupervisedExperimentTrainer`| accuracy, macro_f1, weighted_f1, per-class breakdowns    |
| `ssl`        | `SimCLRExperimentTrainer`    | val_loss, val_contrastive_acc, per-epoch contrastive acc |

**Auto-detection:** Suites whose name starts with `ssl_` are automatically
treated as SSL mode. For all other suites, supervised mode is used. Override
with `--mode supervised` or `--mode ssl` if needed.

## Available Sweep Suites

### Supervised Suites

| Suite                | Runs | Description                                             | Est. Time   |
|----------------------|------|---------------------------------------------------------|-------------|
| `quick_baseline`     | 5    | Learning rate sensitivity: 1e-5 → 1e-3                 | ~50 min     |
| `standard_baseline`  | 60   | LR × batch size × dropout grid                         | ~10 hours   |
| `comprehensive`      | 3600 | Full architecture search (embed_dim, layers, heads)    | Several days|
| `optimization_focus` | 36   | LR + weight decay, warmup, scheduler, mixed precision  | ~6 hours    |
| `scheduler_ablation` | 27   | Cosine warm-restarts + other scheduler comparisons     | ~4.5 hours  |

#### `quick_baseline` (5 configs)
Best for verifying the pipeline end-to-end before committing to long sweeps.

#### `standard_baseline` (60 configs)
- Learning rates: `[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]`
- Batch sizes: `[16, 32, 64]`
- Dropout rates: `[0.0, 0.1, 0.2, 0.3]`

#### `comprehensive` (3,600 configs)
All of `standard_baseline` plus architecture search:
- Embedding dimensions: `[128, 256, 512]`
- Number of layers: `[3, 6, 12]`
- Number of heads: `[4, 8, 16]`

#### `optimization_focus` (36 configs)
Focused on training dynamics:
- LR + weight decay combinations
- Scheduler types: constant, cosine, linear_decay, reduce_on_plateau, cosine_warm_restarts
- Warmup steps: `[0, 500, 1000]`
- Mixed precision: on/off

#### `scheduler_ablation` (27 configs)
Deep-dive into scheduling strategies:
- Cosine warm-restarts with varying warmup (250–2000 steps) and minimum LR
- Cross-scheduler comparison with different warmup durations

### SSL Suites

| Suite                | Runs | Description                                              | Est. Time   |
|----------------------|------|----------------------------------------------------------|-------------|
| `ssl_sanity`         | 1    | Single quick config to verify SSL pipeline               | ~10 min     |
| `ssl_lr`             | 4    | Learning rate sweep for SimCLR pretraining               | ~40 min     |
| `ssl_standard`       | 20   | LR × temperature × weight decay                          | ~3.3 hours  |
| `ssl_comprehensive`  | 720  | Full SSL sweep (LR, temp, projection, batch, aug, decay) | ~5 days     |

#### `ssl_sanity` (1 config)
Single run: `lr=3e-4, batch=16, temp=0.07`. Use this to verify your SSL
pipeline before launching longer sweeps.

#### `ssl_lr` (4 configs)
Learning rates: `[1e-4, 3e-4, 5e-4, 1e-3]`

#### `ssl_standard` (20 configs)
- Learning rates: `[1e-4, 3e-4, 5e-4, 1e-3]`
- Temperature: `[0.03, 0.05, 0.07, 0.1, 0.2]`
- Weight decay: `[0.0, 1e-6, 1e-5, 1e-4, 1e-3]`

#### `ssl_comprehensive` (720 configs)
All of `ssl_standard` plus:
- Projection head hidden dim: `[128, 256, 512]`
- Projection head output dim: `[64, 128, 256]`
- Batch sizes: `[16, 32, 64, 128]`
- Augmentation strength: probability `[0.3–0.9]`, freq/time masking params

## Per-Run Seeding

Each run uses a deterministic seed derived from the base seed:

```
run_seed = base_seed + run_index
```

This ensures:
- Each run has distinct randomness (different data shuffles, augmentation draws)
- Runs are fully reproducible from their seed alone
- The base seed is logged in the experiment summary

## Data Distillation

For fast iteration, you can subsample the dataset via config:

```yaml
data:
  num_classes: 20              # Use only the top 20 classes by sample count
  num_samples_per_class: 50    # Take 50 samples per class
```

When both values are set, the runner:
1. Ranks classes by sample count (descending)
2. Selects the top `num_classes` that have at least `num_samples_per_class` samples
3. Randomly samples `num_samples_per_class` from each
4. Shuffles the result

If neither value is set, the full dataset is used.

## Results Analysis

### `results.csv`

Aggregate metrics for all runs in one CSV. Columns include:

**Always present:**
- `run_id`, `sweep_name`, `timestamp`

**Sweep hyperparameters** (dotted-path keys):
- e.g. `training.learning_rate`, `training.batch_size`, `model.dropout`

**Supervised metrics:**
- `accuracy`, `macro_f1`, `weighted_f1`

**SSL metrics:**
- `val_loss`, `val_contrastive_acc`

**On error:**
- `error` (message), `error_traceback` (full traceback)

### `EXPERIMENT_SUMMARY.md`

Auto-generated summary at the end of a sweep. Contains:
- Experiment ID, mode, timestamp, total runs, base seed
- Top 5 runs ranked by the mode-appropriate metric
  - Supervised: sorted by accuracy (highest first)
  - SSL: sorted by val_loss (lowest first)
- Next-step recommendations

### Per-Run Artifacts

Each `run_XXXX_sweep_name/` directory contains:

| File                        | Description                                         |
|-----------------------------|-----------------------------------------------------|
| `config.yaml`               | Exact config used — **always check this**           |
| `best_model.pth`            | Best checkpoint (encoder-only for SSL)              |
| `label_mappings.json`       | Class name ↔ integer index mapping                  |
| `training_metrics.json`     | Epoch-by-epoch loss, accuracy, etc.                 |
| `evaluation_metrics.json`   | Final test metrics with per-class breakdowns        |
| `EVALUATION_REPORT.md`      | Human-readable report with key findings             |
| `confusion_matrix.png`      | Confusion matrix heatmap                            |
| `per_class_metrics.png`     | Precision/recall/F1 bar chart per class             |

**Note:** SSL checkpoints (`best_model.pth`) contain encoder weights only
(the projection head is excluded) so they can be used directly for linear
probing or fine-tuning.

## Workflow: Baseline → Contrastive → Comparison

### Phase 1 — Supervised Baseline
1. `--suite standard_baseline` (or `comprehensive` if compute allows)
2. Inspect `results.csv` — sort by accuracy/macro_f1
3. Identify best hyperparameters; document in `BASELINE_CONFIG.md`
4. Save the best run's checkpoint

### Phase 2 — SSL Pretraining
1. `--suite ssl_sanity` to verify pipeline
2. `--suite ssl_standard` for meaningful sweep
3. Review `results.csv` — sort by val_loss
4. Best encoder checkpoint → linear probing evaluation

### Phase 3 — Comparison
1. Compare supervised baseline vs. SSL linear probe
2. Generate comparison plots and tables
3. Document findings

## Extending the Framework

### Adding a New Sweep

Edit `experiments/sweep_configs.py`:

```python
MY_SWEEP = HyperparameterSweep(
    name="my_custom_sweep",
    description="Tests something specific",
    params={
        "training.learning_rate": [1e-4, 5e-4],
        "model.embed_dim": [128, 256],
    },
)

SWEEP_SUITES["my_suite"] = [MY_SWEEP]
```

Then run:
```bash
python -m experiments.experiment_runner --suite my_suite
```

Sweep parameters use **dotted config paths** matching the YAML structure
(e.g. `training.learning_rate`, `augmentation.prob`,
`training.mixed_precision.enabled`). These are applied via
`set_nested_config()` — no mapping dictionary is needed.

### Adding a New SSL Sweep Suite

Prefix the suite name with `ssl_` so auto-detection works:

```python
SWEEP_SUITES["ssl_my_experiment"] = [MY_SSL_SWEEP]
```

Or explicitly pass `--mode ssl`.

### Adding New Metrics

Edit `src/evaluation/metrics_collector.py` → `compute_metrics()`:

```python
def compute_metrics(self) -> Dict[str, Any]:
    # ... existing code ...
    my_metric = compute_my_metric(all_preds, all_labels)
    self.metrics["my_metric"] = my_metric
    return self.metrics
```

## Configuration Reference

### Base Config: `configs/config.yaml`

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  device: "cuda"
  weight_decay: 0.0
  warmup_steps: 0
  scheduler_type: "constant"         # constant | cosine | linear_decay | reduce_on_plateau | cosine_warm_restarts
  min_lr: 1e-6                       # Floor LR for cosine scheduler
  mixed_precision:
    enabled: false                   # AMP training (~2x speedup on modern GPUs)
  temperature: 0.07                  # Contrastive temperature (SSL only)

model:
  embed_dim: 256
  num_layers: 6
  heads: 8
  dropout: 0.1

augmentation:
  enabled: true
  prob: 0.5
  freq_mask_param: 6
  time_mask_param: 10

projection:                          # SSL projection head
  hidden_dim: 256
  output_dim: 128

data:
  num_classes: null                  # null = use all classes
  num_samples_per_class: null        # null = use all samples
  label_column: "label"             # Column name in CSV

logging:
  wandb_project: "bird-song-classifier"
```

## Reproducibility

- **Base seed** is set via `--seed` (default: 42)
- **Per-run seed** = `base_seed + run_index` for distinct but deterministic runs
- Seeds are set for `random`, `numpy`, and `torch` (including CUDA)
- Full config is saved per-run in `config.yaml`
- To reproduce a specific run, use the saved `config.yaml` directly

## Troubleshooting

### Out of Memory (OOM)
- Reduce `training.batch_size`
- Reduce `model.embed_dim` or `model.num_layers`
- Enable `training.mixed_precision.enabled: true`

### Slow Training
- Increase `training.batch_size`
- Enable mixed precision for ~2x speedup
- Use data distillation (`num_classes` / `num_samples_per_class`) for faster iteration
- Reduce `model.num_layers`

### Results Not Improving
- Check learning rate — too high causes loss oscillation, too low stalls progress
- Check augmentation strength — `augmentation.prob` might be too aggressive
- Increase model capacity (`embed_dim`, `num_layers`)
- For SSL: check temperature and batch size (more negatives generally helps)

### SSL Pipeline Issues
- Start with `ssl_sanity` suite — a single run with known-good params
- Ensure batch size ≥ 16 (contrastive learning needs negative samples)
- Check that augmentation produces meaningfully different views

## Next Steps

1. ✅ Run `quick_baseline` to verify the supervised pipeline works
2. ✅ Run `standard_baseline` to establish baseline metrics
3. ✅ Review `results.csv` to find the best supervised config
4. ✅ Run `ssl_sanity` to verify the SSL pipeline works
5. ✅ Run `ssl_standard` to find the best contrastive pretraining config
6. ⏭️ Linear-probe the best SSL encoder against the supervised baseline
```
