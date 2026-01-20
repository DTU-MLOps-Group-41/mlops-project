"""PyTorch Lightning DataModule for customer support ticket classification."""

from pathlib import Path

import lightning.pytorch as pl
from datasets import Dataset
from torch.utils.data import DataLoader

from customer_support.data import SEED, load_parquet_dataset


class TicketDataModule(pl.LightningDataModule):
    """Lightning DataModule for customer support ticket classification.

    Loads preprocessed parquet files directly using paths from Hydra config.

    Args:
        train_path: Path to training parquet file.
        val_path: Path to validation parquet file.
        test_path: Path to test parquet file.
        batch_size: Batch size for DataLoaders (default: 32).
        num_workers: Number of workers for DataLoaders (default: 0).
        seed: Random seed for reproducibility (default: 42).

    Example:
        >>> datamodule = TicketDataModule(
        ...     train_path="data/preprocessed/small_train.parquet",
        ...     val_path="data/preprocessed/small_validation.parquet",
        ...     test_path="data/preprocessed/small_test.parquet",
        ...     batch_size=32,
        ... )
        >>> datamodule.setup(stage="fit")
        >>> train_loader = datamodule.train_dataloader()
        >>> for batch in train_loader:
        ...     input_ids = batch["input_ids"]
        ...     labels = batch["labels"]
    """

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Will be populated in setup()
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Either "fit", "validate", "test", or "predict"
        """
        if stage == "fit" or stage is None:
            self.train_dataset = load_parquet_dataset(self.train_path)
            self.val_dataset = load_parquet_dataset(self.val_path)

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = load_parquet_dataset(self.val_path)

        if stage == "test" or stage is None:
            self.test_dataset = load_parquet_dataset(self.test_path)

        if stage == "predict":
            self.test_dataset = load_parquet_dataset(self.test_path)

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return predict DataLoader (uses test dataset)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
