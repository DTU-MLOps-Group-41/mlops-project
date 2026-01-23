# CLI Commands

This project uses [Invoke](https://www.pyinvoke.org/) for task automation. All commands are defined in `tasks.py`.

## Usage

Run commands with:

```bash
invoke <command>
# or
inv <command>
```

List all available commands:

```bash
invoke --list
```

## Data Commands

### `download-data`

Download raw dataset from Kaggle.

```bash
invoke download-data
```

Downloads all dataset sizes (small, medium, full) to `data/raw/`.

**Prerequisites:** Kaggle credentials configured ([setup guide](https://www.kaggle.com/docs/api#authentication))

### `preprocess-data`

Preprocess datasets (clean, tokenize, split, save as Parquet).

```bash
# Process all dataset sizes
invoke preprocess-data

# Process a specific size
invoke preprocess-data --dataset-type small
invoke preprocess-data --dataset-type medium
invoke preprocess-data --dataset-type full

# With sequence length filtering
invoke preprocess-data --length-percentile 90 --length-handling trim
invoke preprocess-data --length-percentile 95 --length-handling drop
```

**Options:**

| Option | Description |
|--------|-------------|
| `--dataset-type` | Dataset size: `small`, `medium`, or `full` |
| `--length-percentile` | Percentile threshold for sequence length (0 = no filtering) |
| `--length-handling` | How to handle long sequences: `trim` or `drop` |

## Training Commands

### `train`

Train the model with default configuration.

```bash
invoke train
```

This runs `uv run src/customer_support/train.py`. For configuration options, see [Training](training.md).

### `evaluate`

Evaluate a trained model on the test set.

```bash
invoke evaluate --checkpoint path/to/model.ckpt
```

**Required:** `--checkpoint` - Path to model checkpoint file

### `visualize`

Generate visualizations (confusion matrix) for a trained model.

```bash
invoke visualize --checkpoint path/to/model.ckpt
```

**Required:** `--checkpoint` - Path to model checkpoint file

## Testing Commands

### `test`

Run the test suite with coverage reporting.

```bash
invoke test
```

This runs:
1. `uv run coverage run -m pytest tests/`
2. `uv run coverage report -m -i`

## Docker Commands

### `docker-build`

Build both API and training Docker images.

```bash
invoke docker-build

# With verbose progress output
invoke docker-build --progress auto
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--progress` | `plain` | Docker build progress output: `plain`, `auto`, `tty` |

**Images built:**

- `train:latest` - Training image from `dockerfiles/train.dockerfile`
- `api:latest` - API image from `dockerfiles/api.dockerfile`

## Documentation Commands

### `build-docs`

Build the MkDocs documentation site.

```bash
invoke build-docs
```

Output is written to `docs/build/`.

### `serve-docs`

Start the documentation development server.

```bash
invoke serve-docs
```

Opens at [http://127.0.0.1:8000](http://127.0.0.1:8000) with live reload.

## Utility Commands

### `project-tree`

Generate a tree view of the project structure.

```bash
# Full tree
invoke project-tree

# Limit depth
invoke project-tree --depth 2
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--depth` | 0 (unlimited) | Maximum directory depth to display |

## Quick Reference

| Command | Description |
|---------|-------------|
| `invoke download-data` | Download Kaggle dataset |
| `invoke preprocess-data` | Preprocess all datasets |
| `invoke train` | Train model |
| `invoke evaluate --checkpoint <path>` | Evaluate model |
| `invoke visualize --checkpoint <path>` | Generate visualizations |
| `invoke test` | Run tests with coverage |
| `invoke docker-build` | Build Docker images |
| `invoke build-docs` | Build documentation |
| `invoke serve-docs` | Serve documentation locally |
