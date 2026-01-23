# Getting Started

This guide will help you set up the project for local development.

## Prerequisites

- **Python 3.12** - Check with `python --version`
- **UV** - Modern Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Git** - Version control
- **Docker** (optional) - For containerized training/inference

For cloud features:
- **Google Cloud SDK** - For DVC data access and Vertex AI training
- **Kaggle account** - For downloading the dataset

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd mlops-project
```

### 2. Install dependencies

Using UV with CPU-only PyTorch (recommended for most development):

```bash
uv sync --extra cpu
```

For CUDA 12.8 GPU support:

```bash
uv sync --extra cu128
```

### 3. Set up pre-commit hooks

```bash
uv run pre-commit install
```

This enables automatic code quality checks (Ruff linting/formatting, MyPy type checking, Gitleaks secret scanning) before each commit.

## Data Setup

### Option A: Pull pre-processed data with DVC (recommended)

If you have GCP access configured:

```bash
uv run dvc pull
```

This downloads the pre-processed datasets from Google Cloud Storage.

### Option B: Download and preprocess from Kaggle

1. Set up Kaggle credentials ([instructions](https://www.kaggle.com/docs/api#authentication))

2. Download raw data:

```bash
invoke download-data
```

3. Preprocess the data:

```bash
# Process all dataset sizes (small, medium, full)
invoke preprocess-data

# Or process a specific size
invoke preprocess-data --dataset-type medium
```

## Verify Setup

Run the test suite to verify everything is working:

```bash
invoke test
```

Run a quick training to test the full pipeline:

```bash
uv run python -m customer_support.train training=baseline dataset=small
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | Weights & Biases API key for experiment tracking | For training |
| `KAGGLE_USERNAME` | Kaggle username | For data download |
| `KAGGLE_KEY` | Kaggle API key | For data download |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON | For DVC/cloud |

To disable W&B logging during local development:

```bash
export WANDB_MODE=disabled
```

## Next Steps

- [Data Processing](data.md) - Learn about the dataset and preprocessing pipeline
- [Training](training.md) - Train models locally or on the cloud
- [CLI Commands](cli.md) - All available invoke tasks
