from pathlib import Path
from typing import Callable, Optional

import kagglehub
import pandas as pd
import torch
import torch.utils.data
import typer
from datasets import Dataset, DatasetDict
from loguru import logger
from transformers import DistilBertTokenizer

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/preprocessed")
SEED = 42
LABEL_MAP = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
DATASET_FILES = {
    "full": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "medium": "dataset-tickets-multi-lang-4-20k.csv",
    "small": "dataset-tickets-multi-lang3-4k.csv",
}
VALID_SPLITS = ["train", "validation", "test"]
VALID_DATASET_TYPES = ["small", "medium", "full"]


# ============================================================================
# MAIN DATASET CLASS
# ============================================================================


class TicketDataset(torch.utils.data.Dataset):
    """Customer support ticket dataset for priority classification.

    Following torchvision.datasets.MNIST design patterns for consistency with
    PyTorch ecosystem. Supports loading preprocessed data or preprocessing from raw CSV.

    Args:
        root: Root directory where dataset files are stored (e.g., "data")
        split: Dataset split - "train", "validation", or "test" (default: "train")
        dataset_type: Dataset size - "small", "medium", or "full" (default: "small")
        download: If True, download and preprocess if not found (default: False)
        force_preprocess: If True, force reprocessing even if data exists (default: False)
        transform: Optional transform applied on tokenized samples
        target_transform: Optional transform applied on labels
        model_name: Tokenizer model name (default: "distilbert-base-multilingual-cased")

    Raises:
        RuntimeError: If data not found and download=False
        ValueError: If split or dataset_type is invalid

    Example:
        >>> # Load preprocessed training data (most common)
        >>> train_data = TicketDataset(root="data", split="train", dataset_type="small")
        >>>
        >>> # Auto-preprocess if needed
        >>> train_data = TicketDataset(root="data", split="train", download=True)
        >>>
        >>> # Force reprocessing
        >>> train_data = TicketDataset(root="data", split="train", force_preprocess=True)
        >>>
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(train_data, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     input_ids = batch["input_ids"]
        ...     labels = batch["labels"]
        ...     # Training logic here
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        dataset_type: str = "small",
        download: bool = False,
        force_preprocess: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        model_name: str = "distilbert-base-multilingual-cased",
    ) -> None:
        """Initialize TicketDataset."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got: {split}")
        if dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(f"dataset_type must be one of {VALID_DATASET_TYPES}, got: {dataset_type}")

        self.root = Path(root)
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform
        self.model_name = model_name
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "preprocessed"

        if force_preprocess or (not self._check_exists() and download):
            self._preprocess()
        elif not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.processed_dir / self.dataset_type}.\n\n"
                f"Set download=True to auto-download and preprocess:\n"
                f"  TicketDataset(root='{self.root}', split='{self.split}', download=True)"
            )

        self._dataset = self._load_data()
        logger.info(f"TicketDataset initialized: {len(self)} samples ({self.dataset_type}/{self.split})")

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys: 'input_ids', 'attention_mask', 'labels'
            All values are torch.Tensor

        Notes:
            Applies self.transform if provided
            Applies self.target_transform to labels if provided
        """
        sample = self._dataset[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            sample["labels"] = self.target_transform(sample["labels"])

        return sample

    def __repr__(self) -> str:
        """String representation showing configuration."""
        return (
            f"{self.__class__.__name__}(split={self.split}, dataset_type={self.dataset_type}, num_samples={len(self)})"
        )

    @classmethod
    def get_label_map(cls) -> dict[str, int]:
        """Return the label mapping used for encoding priorities.

        Returns:
            Dictionary mapping priority strings to integer labels
        """
        return LABEL_MAP.copy()

    @staticmethod
    def _download_kaggle_dataset() -> Path:
        """Download the multilingual customer support tickets dataset from Kaggle.

        Returns:
            Path to the downloaded dataset directory

        Notes:
            Uses KaggleHub caching (downloads only if not cached)
            Downloads all dataset files at once
        """
        logger.info("Downloading dataset from Kaggle (using KaggleHub cache)...")
        data_path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")
        logger.info(f"Dataset available at: {data_path}")
        return Path(data_path)

    @staticmethod
    def _save_raw_csv(kaggle_path: Path, dataset_type: str, output_dir: Path = RAW_DATA_DIR) -> Path:
        """Copy specific CSV file from KaggleHub cache to raw data directory.

        Args:
            kaggle_path: Path to KaggleHub dataset directory
            dataset_type: One of "small", "medium", "full"
            output_dir: Directory to save raw CSV

        Returns:
            Path to saved CSV file

        Raises:
            ValueError: If dataset_type is invalid
            FileNotFoundError: If source CSV file doesn't exist in KaggleHub cache
        """
        if dataset_type not in DATASET_FILES:
            raise ValueError(f"dataset_type must be one of {list(DATASET_FILES.keys())}, got: {dataset_type}")

        source_filename = DATASET_FILES[dataset_type]
        source_path = kaggle_path / source_filename

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_type}.csv"

        df = pd.read_csv(source_path)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {dataset_type} dataset to: {output_path}")
        return output_path

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw dataframe by selecting relevant columns and handling missing data.

        Args:
            df: Raw dataframe with all columns

        Returns:
            Cleaned dataframe with 'body' and 'priority' columns

        Notes:
            Selects only 'body' and 'priority' columns
            Drops rows with missing values
            Converts body to string type
        """
        data = df[["body", "priority"]].copy()
        data = data.dropna(subset=["body", "priority"])
        data["body"] = data["body"].astype(pd.StringDtype())
        data["priority"] = data["priority"].astype("category")
        logger.info(f"Cleaned data: {len(data)} rows remaining after dropping missing values")
        return data

    @staticmethod
    def _encode_labels(df: pd.DataFrame, label_map: dict[str, int] = LABEL_MAP) -> pd.DataFrame:
        """Encode priority labels as integers.

        Args:
            df: Dataframe with 'priority' column
            label_map: Mapping from priority string to integer

        Returns:
            Dataframe with additional 'labels' column (int)

        Notes:
            Converts priority to lowercase before mapping
            Drops rows where label mapping failed
        """
        data = df.copy()
        data["labels"] = data["priority"].str.lower().map(label_map)

        rows_before = len(data)
        data = data.dropna(subset=["labels"])
        rows_after = len(data)

        if rows_before != rows_after:
            logger.warning(f"Dropped {rows_before - rows_after} rows with unknown priority labels")

        data["labels"] = data["labels"].astype(int)
        logger.info(f"Encoded labels: {data['labels'].value_counts().to_dict()}")
        return data

    @staticmethod
    def _tokenize_dataset(dataset: Dataset, model_name: str = "distilbert-base-multilingual-cased") -> Dataset:
        """Tokenize text data using DistilBERT tokenizer.

        Args:
            dataset: HuggingFace Dataset with 'body' column
            model_name: Tokenizer model name

        Returns:
            Tokenized dataset with input_ids, attention_mask

        Notes:
            Removes 'body' column after tokenization
            Removes '__index_level_0__' if present
            Sets format to 'torch'
        """
        logger.info(f"Tokenizing with {model_name}...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["body"], truncation=True, padding=False)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        cols_to_remove = ["body"]
        if "__index_level_0__" in tokenized_dataset.column_names:
            cols_to_remove.append("__index_level_0__")

        tokenized_dataset = tokenized_dataset.remove_columns(cols_to_remove)
        tokenized_dataset.set_format("torch")

        logger.info(f"Tokenization complete. Columns: {tokenized_dataset.column_names}")
        return tokenized_dataset

    @staticmethod
    def _split_dataset(
        dataset: Dataset, test_size: float = 0.2, val_size: float = 0.5, seed: int = SEED
    ) -> DatasetDict:
        """Split dataset into train/validation/test sets.

        Args:
            dataset: Tokenized dataset with 'labels' column
            test_size: Proportion of data for test+validation (default: 0.2)
            val_size: Proportion of test+validation for validation (default: 0.5)
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with 'train', 'validation', 'test' splits

        Notes:
            Default split: 80% train, 10% validation, 10% test
            Uses stratified splitting by labels
            Falls back to non-stratified split if stratification fails (very small datasets)
        """
        logger.info("Splitting dataset...")

        # Cast labels to ClassLabel type for stratified splitting
        from datasets import ClassLabel

        num_classes = len(set(dataset["labels"]))
        dataset = dataset.cast_column("labels", ClassLabel(num_classes=num_classes))

        try:
            split_dataset_train_test = dataset.train_test_split(
                test_size=test_size, seed=seed, shuffle=True, stratify_by_column="labels"
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), using non-stratified split")
            split_dataset_train_test = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)

        try:
            test_valid = split_dataset_train_test["test"].train_test_split(
                test_size=val_size, seed=seed, shuffle=True, stratify_by_column="labels"
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), using non-stratified split")
            test_valid = split_dataset_train_test["test"].train_test_split(test_size=val_size, seed=seed, shuffle=True)

        final_dataset = DatasetDict(
            {
                "train": split_dataset_train_test["train"],
                "validation": test_valid["train"],
                "test": test_valid["test"],
            }
        )

        logger.info(
            f"Split sizes: Train={len(final_dataset['train'])}, Val={len(final_dataset['validation'])}, "
            f"Test={len(final_dataset['test'])}"
        )

        return final_dataset

    def _check_exists(self) -> bool:
        """Check if preprocessed data exists for current configuration.

        Returns:
            True if Parquet file for the split exists
        """
        parquet_file = self.processed_dir / f"{self.dataset_type}_{self.split}.parquet"
        return parquet_file.exists()

    def _download_raw_data(self) -> Path:
        """Download raw CSV from Kaggle if not present.

        Returns:
            Path to raw CSV file

        Notes:
            Uses KaggleHub caching (idempotent)
            Only downloads if raw CSV doesn't exist
        """
        csv_path = self.raw_dir / f"{self.dataset_type}.csv"

        if csv_path.exists():
            logger.info(f"Raw CSV already exists: {csv_path}")
            return csv_path

        logger.info("Downloading from Kaggle...")
        kaggle_path = self._download_kaggle_dataset()
        return self._save_raw_csv(kaggle_path, self.dataset_type, self.raw_dir)

    def _preprocess(self) -> None:
        """Run complete preprocessing pipeline: clean → encode → tokenize → split → save.

        Raises:
            FileNotFoundError: If raw data cannot be obtained
        """
        logger.info(f"{'=' * 80}")
        logger.info(f"Preprocessing {self.dataset_type} dataset...")
        logger.info(f"{'=' * 80}")

        csv_path = self._download_raw_data()

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"Empty CSV file: {csv_path}")

        logger.info(f"Loaded {len(df)} rows from {csv_path}")

        df = self._clean_dataframe(df)
        df = self._encode_labels(df)

        hf_dataset = Dataset.from_pandas(df[["body", "labels"]])
        tokenized_dataset = self._tokenize_dataset(hf_dataset, self.model_name)
        dataset_dict = self._split_dataset(tokenized_dataset)

        # Save each split as a separate Parquet file
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for split_name in VALID_SPLITS:
            output_file = self.processed_dir / f"{self.dataset_type}_{split_name}.parquet"
            dataset_dict[split_name].to_parquet(output_file)
            logger.info(f"Saved {split_name} split to: {output_file}")

        logger.success(f"Preprocessed dataset saved as Parquet files in: {self.processed_dir}")

    def _load_data(self) -> Dataset:
        """Load preprocessed split from disk into memory.

        Returns:
            HuggingFace Dataset for the specified split

        Raises:
            FileNotFoundError: If split data not found
        """
        parquet_file = self.processed_dir / f"{self.dataset_type}_{self.split}.parquet"

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_file}\nAvailable files: {list(self.processed_dir.glob('*.parquet'))}"
            )

        dataset = Dataset.from_parquet(str(parquet_file))
        dataset.set_format("torch")

        logger.info(f"Loaded {len(dataset)} samples from {parquet_file}")
        return dataset


# ============================================================================
# CLI ENTRY POINTS
# ============================================================================

app = typer.Typer(help="Customer support ticket data pipeline")


@app.command(name="download")
def download_command() -> None:
    """Download all dataset sizes from Kaggle to data/raw/.

    Always downloads fresh copies from KaggleHub cache, overwriting existing files.
    KaggleHub uses its own cache, but files are copied fresh to data/raw/.

    Example:
        uv run src/customer_support/data.py download
    """
    for dtype in VALID_DATASET_TYPES:
        try:
            logger.info(f"Downloading {dtype} dataset...")
            kaggle_path = TicketDataset._download_kaggle_dataset()
            TicketDataset._save_raw_csv(kaggle_path, dtype, RAW_DATA_DIR)
        except Exception as e:
            logger.error(f"Error downloading {dtype}: {e}")


@app.command(name="preprocess")
def preprocess_command(
    dataset_type: str = typer.Option(None, "--dataset-type", "-d", help="Dataset size: small, medium, or full"),
    all_datasets: bool = typer.Option(False, "--all", "-a", help="Process all dataset sizes"),
    model_name: str = typer.Option("distilbert-base-multilingual-cased", "--model", "-m", help="Tokenizer model name"),
) -> None:
    """Preprocess datasets: clean, tokenize, split, and save.

    Always forces reprocessing, even if preprocessed data already exists.

    Examples:
        # Process single dataset
        uv run src/customer_support/data.py preprocess -d small

        # Process all datasets
        uv run src/customer_support/data.py preprocess --all

        # Use custom tokenizer
        uv run src/customer_support/data.py preprocess -d medium -m bert-base-multilingual-cased
    """
    if all_datasets:
        for dtype in VALID_DATASET_TYPES:
            _ = TicketDataset(
                root="data", split="train", dataset_type=dtype, force_preprocess=True, model_name=model_name
            )
    elif dataset_type:
        if dataset_type not in VALID_DATASET_TYPES:
            raise typer.BadParameter("dataset_type must be one of: small, medium, full")
        _ = TicketDataset(
            root="data", split="train", dataset_type=dataset_type, force_preprocess=True, model_name=model_name
        )
    else:
        raise typer.BadParameter("Must specify either --dataset-type or --all")


if __name__ == "__main__":
    app()
