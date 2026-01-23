# Training

This guide covers how to train models locally. For cloud training on Vertex AI, see [Cloud Infrastructure](cloud.md).

## Basic Training

Run training with default configuration (medium dataset, baseline training):

```bash
uv run python -m customer_support.train
```

Or using invoke:

```bash
invoke train
```

## Configuration System

Training uses [Hydra](https://hydra.cc/) for configuration management. Configs are located in `configs/` with this structure:

```
configs/
├── config.yaml          # Main config (defaults, paths, wandb)
├── dataset/
│   ├── small.yaml       # ~4k samples
│   ├── medium.yaml      # ~20k samples (default)
│   └── full.yaml        # ~50k+ samples
├── training/
│   ├── baseline.yaml    # Quick experiments (default)
│   ├── production.yaml  # Production-quality training
│   └── best_sweep.yaml  # Optimized hyperparameters
└── experiment/
    ├── baseline.yaml    # Baseline experiment preset
    ├── production.yaml  # Production experiment preset
    └── best.yaml        # Best hyperparameters from sweep
```

## Configuration Options

### Dataset Selection

```bash
# Small dataset for quick tests
uv run python -m customer_support.train dataset=small

# Full dataset for production
uv run python -m customer_support.train dataset=full
```

### Training Configurations

**Baseline** (default) - Fast experiments:
```bash
uv run python -m customer_support.train training=baseline
```
- 5 epochs, batch size 64, learning rate 1e-4
- Early stopping patience: 3

**Production** - Full training:
```bash
uv run python -m customer_support.train training=production
```
- 10 epochs, batch size 32, learning rate 5e-5
- Cosine LR scheduler with warmup
- Gradient clipping, gradient accumulation
- Deterministic training

### Experiment Presets

Use experiment configs for predefined combinations:

```bash
# Best hyperparameters from sweep + full dataset
uv run python -m customer_support.train experiment=best
```

### Overriding Parameters

Override any parameter from command line:

```bash
uv run python -m customer_support.train \
    dataset=medium \
    training.batch_size=32 \
    training.learning_rate=5e-5 \
    training.num_epochs=10
```

### Hardware Configuration

```bash
# Force CPU
uv run python -m customer_support.train accelerator=cpu

# Specific GPU
uv run python -m customer_support.train devices=1 accelerator=gpu
```

## Experiment Tracking

Training is automatically logged to Weights & Biases (if `WANDB_API_KEY` is set).

### W&B Configuration

In `configs/config.yaml`:

```yaml
wandb:
  entity: "dtu-mlops-g41"
  project: "customer_support"
  mode: "online"       # online, offline, disabled
  log_model: true      # Log model checkpoints
```

### Disable W&B Logging

```bash
# Via environment variable
export WANDB_MODE=disabled

# Or via config override
uv run python -m customer_support.train wandb.mode=disabled
```

## Training Output

Training creates output directories under `outputs/YYYY-MM-DD/HH-MM-SS/`:

```
outputs/2024-01-15/10-30-00/
├── .hydra/              # Hydra config snapshots
├── checkpoints/         # Model checkpoints
│   └── epoch=X-step=Y.ckpt
├── csv_logs/            # Training metrics CSV
└── train.log            # Training logs
```

## Callbacks

### Early Stopping
Stops training when validation accuracy stops improving:
- Monitor: `val_accuracy`
- Patience: configurable (default: 3 epochs)

### Model Checkpointing
Saves best model(s) based on validation accuracy:
- Monitor: `val_accuracy`
- Mode: `max`
- Save top K: configurable (default: 1)

## Key Training Parameters

| Parameter | Baseline | Production | Description |
|-----------|----------|------------|-------------|
| `batch_size` | 64 | 32 | Batch size per device |
| `learning_rate` | 1e-4 | 5e-5 | Initial learning rate |
| `num_epochs` | 5 | 10 | Maximum epochs |
| `patience` | 3 | 5 | Early stopping patience |
| `precision` | 16-mixed | 16-mixed | Mixed precision training |
| `gradient_clip_val` | null | 1.0 | Gradient clipping |
| `accumulate_grad_batches` | 1 | 2 | Gradient accumulation |

## Example: Full Training Run

```bash
# Production training on full dataset with W&B logging
export WANDB_API_KEY=your_key_here

uv run python -m customer_support.train \
    experiment=best \
    seed=42
```
