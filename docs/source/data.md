## Data Processing

This project uses the [Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) dataset from Kaggle for training a ticket priority classification model. The dataset contains 28,600 customer support tickets with descriptions and priority labels.

Three dataset sizes are available:
- **small**: ~4,000 tickets (for rapid experimentation)
- **medium**: ~20,000 tickets (for balanced training)
- **full**: ~50,000+ tickets (for maximum performance)

## Dataset Details

The dataset consists of multilingual customer support tickets with the following characteristics:

- **Source**: [Kaggle - Customer IT Support Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
- **Total observations**: 28,600 tickets
- **Features used**:
  - `body`: Customer's ticket description (text)
  - `priority`: Ticket priority level (label)
- **Label mapping**:
  - `very_low` → 0
  - `low` → 1
  - `medium` → 2
  - `high` → 3
  - `critical` → 4
- **Data splits**: 80% train, 10% validation, 10% test (stratified by labels for balanced representation)
- **Tokenization**: DistilBERT multilingual model (`distilbert-base-multilingual-cased`)

Note: The original dataset contains additional fields (topic, department) that are not currently used in the preprocessing pipeline.

## Preprocessing Pipeline

The automated preprocessing pipeline performs the following steps:

1. **Download**: Fetch dataset from Kaggle using KaggleHub (with automatic caching)
2. **Clean**: Select relevant columns (`body`, `priority`) and remove rows with missing values
3. **Encode**: Convert priority strings to integer labels using the label mapping
4. **Tokenize**: Apply DistilBERT multilingual tokenizer to ticket body text
5. **Split**: Divide into train/validation/test sets with stratified sampling
6. **Save**: Export as Parquet files for efficient loading

All preprocessing is handled automatically by the `TicketDataset` class or can be run manually via CLI commands.

## Quick Start

### Download raw data from Kaggle

```bash
uv run src/customer_support/data.py download
```

Or using invoke:

```bash
uv run invoke download-data
```

### Preprocess a dataset

```bash
# Preprocess the small dataset
uv run src/customer_support/data.py preprocess -d small

# Preprocess all datasets
uv run src/customer_support/data.py preprocess --all
```

Or using invoke:

```bash
uv run invoke preprocess-data --dataset-type small
uv run invoke preprocess-data  # processes all sizes
```

### Load preprocessed data in Python

```python
from customer_support.data import TicketDataset

# Load preprocessed training data
train_data = TicketDataset(root="data", split="train", dataset_type="small")

# Access a sample
sample = train_data[0]
print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
```

### Use with PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from customer_support.data import TicketDataset

train_data = TicketDataset(root="data", split="train", dataset_type="small")
loader = DataLoader(train_data, batch_size=32, shuffle=True)

for batch in loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # Training logic here
```

## CLI Commands

The data module provides two main CLI commands via Typer.

### Download Command

Downloads all dataset sizes from Kaggle to `data/raw/`.

```bash
uv run src/customer_support/data.py download
```

Or via invoke task:

```bash
uv run invoke download-data
```

This command:
- Uses KaggleHub's caching mechanism (downloads only if not cached)
- Saves all three dataset sizes (small, medium, full) as CSV files
- Overwrites existing files in `data/raw/`

### Preprocess Command

Preprocesses datasets by cleaning, tokenizing, splitting, and saving as Parquet files.

```bash
# Preprocess a specific dataset size
uv run src/customer_support/data.py preprocess --dataset-type small
uv run src/customer_support/data.py preprocess -d medium

# Preprocess all dataset sizes
uv run src/customer_support/data.py preprocess --all

# Use a custom tokenizer
uv run src/customer_support/data.py preprocess -d small -m bert-base-multilingual-cased
```

Or via invoke tasks:

```bash
uv run invoke preprocess-data --dataset-type small
uv run invoke preprocess-data  # processes all sizes
```

Options:
- `-d, --dataset-type`: Dataset size (`small`, `medium`, or `full`)
- `-a, --all`: Process all dataset sizes
- `-m, --model`: Tokenizer model name (default: `distilbert-base-multilingual-cased`)

Note: The preprocess command always forces reprocessing, even if preprocessed data already exists.

## Python API

### TicketDataset Class

The `TicketDataset` class provides a PyTorch-compatible dataset following the design patterns of `torchvision.datasets.MNIST`.

#### Basic Usage

```python
from customer_support.data import TicketDataset

# Load preprocessed data (most common use case)
train_data = TicketDataset(root="data", split="train", dataset_type="small")

# Auto-download and preprocess if data not found
train_data = TicketDataset(root="data", split="train", download=True)

# Force reprocessing even if data exists
train_data = TicketDataset(root="data", split="train", force_preprocess=True)

# Load different splits
val_data = TicketDataset(root="data", split="validation", dataset_type="small")
test_data = TicketDataset(root="data", split="test", dataset_type="small")
```

#### With DataLoader

```python
from torch.utils.data import DataLoader

train_data = TicketDataset(root="data", split="train", dataset_type="small")
loader = DataLoader(train_data, batch_size=32, shuffle=True)

for batch in loader:
    input_ids = batch["input_ids"]           # Shape: [batch_size, seq_length]
    attention_mask = batch["attention_mask"] # Shape: [batch_size, seq_length]
    labels = batch["labels"]                 # Shape: [batch_size]
```

#### Constructor Parameters

- `root` (str | Path): Root directory where dataset files are stored (e.g., `"data"`)
- `split` (str): Dataset split - `"train"`, `"validation"`, or `"test"` (default: `"train"`)
- `dataset_type` (str): Dataset size - `"small"`, `"medium"`, or `"full"` (default: `"small"`)
- `download` (bool): If True, download and preprocess if not found (default: `False`)
- `force_preprocess` (bool): If True, force reprocessing even if data exists (default: `False`)
- `transform` (Callable | None): Optional transform applied to tokenized samples
- `target_transform` (Callable | None): Optional transform applied to labels
- `model_name` (str): Tokenizer model name (default: `"distilbert-base-multilingual-cased"`)

#### Class Methods

```python
# Get the label mapping dictionary
label_map = TicketDataset.get_label_map()
# Returns: {"very_low": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
```

#### Return Format

Each sample returned by `__getitem__` is a dictionary with the following keys:

- `input_ids` (torch.Tensor): Tokenized input IDs, shape `[seq_length]`
- `attention_mask` (torch.Tensor): Attention mask, shape `[seq_length]`
- `labels` (torch.Tensor): Priority label (0-4), shape `[]` (scalar)

#### Custom Transforms

```python
# Apply custom transform to samples
def add_noise(sample):
    sample["input_ids"] = sample["input_ids"] + torch.randn_like(sample["input_ids"]) * 0.01
    return sample

dataset = TicketDataset(root="data", split="train", transform=add_noise)

# Apply custom transform to labels
def offset_labels(label):
    return label + 1

dataset = TicketDataset(root="data", split="train", target_transform=offset_labels)
```

## File Structure

The data directory is organized as follows:

```
data/
├── raw/                          # Raw CSV files from Kaggle
│   ├── small.csv
│   ├── medium.csv
│   └── full.csv
└── preprocessed/                 # Preprocessed Parquet files
    ├── small_train.parquet
    ├── small_validation.parquet
    ├── small_test.parquet
    ├── medium_train.parquet
    ├── medium_validation.parquet
    ├── medium_test.parquet
    ├── full_train.parquet
    ├── full_validation.parquet
    └── full_test.parquet
```

The Parquet format is used for efficient storage and fast loading of preprocessed datasets. Each split is saved as a separate file to enable selective loading.

## Data Versioning

The `data/` directory is tracked with DVC (Data Version Control) to maintain reproducibility across team members and environments.

After preprocessing or modifying data, you should version your changes using DVC. See [Data Version Control](dvc.md) for the complete workflow on:
- Setting up DVC with the DTU databar cluster
- Versioning data changes with `dvc add`, tagging, and pushing
- Pulling specific data versions
- Troubleshooting DVC issues

**Quick reference:**

```bash
# After preprocessing, version the changes
dvc add data/
git add data.dvc .gitignore
git commit -m "Update preprocessed data"
git tag -a "v1.x" -m "data v1.x"
git push --follow-tags
dvc push
```

For detailed instructions and troubleshooting, refer to the [Data Version Control](dvc.md) documentation.
