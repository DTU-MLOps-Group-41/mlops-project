"""PyTorch Lightning DataModule for customer support ticket classification."""

from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from customer_support.data import SEED, TicketDataset


class TicketDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping TicketDataset.

    Provides Lightning integration for the customer support ticket dataset,
    handling data preparation, setup, and DataLoader creation.

    Args:
        root: Root directory for dataset files (default: "data")
        dataset_type: Dataset size - "small", "medium", or "full" (default: "small")
        batch_size: Batch size for DataLoaders (default: 32)
        num_workers: Number of workers for DataLoaders (default: 0)
        download: If True, download and preprocess if not found (default: False)
        force_preprocess: If True, force reprocessing even if data exists (default: False)
        model_name: Tokenizer model name (default: "distilbert-base-multilingual-cased")
        seed: Random seed for reproducibility (default: 42)

    Example:
        >>> datamodule = TicketDataModule(root="data", dataset_type="small", batch_size=32)
        >>> datamodule.setup(stage="fit")
        >>> train_loader = datamodule.train_dataloader()
        >>> for batch in train_loader:
        ...     input_ids = batch["input_ids"]
        ...     labels = batch["labels"]
    """

    def __init__(
        self,
        root: str | Path = "data",
        dataset_type: str = "small",
        batch_size: int = 32,
        num_workers: int = 0,
        download: bool = False,
        force_preprocess: bool = False,
        model_name: str = "distilbert-base-multilingual-cased",
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.force_preprocess = force_preprocess
        self.model_name = model_name
        self.seed = seed

        # Will be populated in setup()
        self.train_dataset: TicketDataset | None = None
        self.val_dataset: TicketDataset | None = None
        self.test_dataset: TicketDataset | None = None

    def prepare_data(self) -> None:
        """Download data if needed (called only on rank 0 in distributed training)."""
        if self.download or self.force_preprocess:
            TicketDataset(
                root=self.root,
                split="train",
                dataset_type=self.dataset_type,
                download=self.download,
                force_preprocess=self.force_preprocess,
                model_name=self.model_name,
            )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Either "fit", "validate", "test", or "predict"
        """
        if stage == "fit" or stage is None:
            self.train_dataset = TicketDataset(
                root=self.root,
                split="train",
                dataset_type=self.dataset_type,
                download=False,
                model_name=self.model_name,
            )
            self.val_dataset = TicketDataset(
                root=self.root,
                split="validation",
                dataset_type=self.dataset_type,
                download=False,
                model_name=self.model_name,
            )

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = TicketDataset(
                root=self.root,
                split="validation",
                dataset_type=self.dataset_type,
                download=False,
                model_name=self.model_name,
            )

        if stage == "test" or stage is None:
            self.test_dataset = TicketDataset(
                root=self.root,
                split="test",
                dataset_type=self.dataset_type,
                download=False,
                model_name=self.model_name,
            )

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
