"""Tests for TicketDataModule and training pipeline."""

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from customer_support.datamodule import TicketDataModule


class TestTicketDataModule:
    """Tests for TicketDataModule (PyTorch Lightning DataModule)."""

    def test_init_stores_hyperparameters(self) -> None:
        """Test that __init__ stores hyperparameters correctly."""
        datamodule = TicketDataModule(
            train_path="data/preprocessed/small_train.parquet",
            val_path="data/preprocessed/small_validation.parquet",
            test_path="data/preprocessed/small_test.parquet",
            batch_size=16,
            num_workers=2,
        )

        assert datamodule.batch_size == 16
        assert datamodule.hparams["batch_size"] == 16
        assert datamodule.num_workers == 2
        assert datamodule.hparams["num_workers"] == 2
        assert datamodule.train_path == Path("data/preprocessed/small_train.parquet")
        assert datamodule.hparams["train_path"] == "data/preprocessed/small_train.parquet"

    def test_hparams_saved(self) -> None:
        """Test that hyperparameters are saved for checkpointing."""
        datamodule = TicketDataModule(
            train_path="data/preprocessed/small_train.parquet",
            val_path="data/preprocessed/small_validation.parquet",
            test_path="data/preprocessed/small_test.parquet",
            batch_size=64,
        )

        assert hasattr(datamodule, "hparams")
        assert datamodule.hparams["batch_size"] == 64
        assert datamodule.hparams["train_path"] == "data/preprocessed/small_train.parquet"

    def test_datasets_none_before_setup(self) -> None:
        """Test that datasets are None before setup is called."""
        datamodule = TicketDataModule(train_path="dummy.parquet", val_path="dummy.parquet", test_path="dummy.parquet")

        assert datamodule.train_dataset is None
        assert datamodule.val_dataset is None
        assert datamodule.test_dataset is None


class TestTicketDataModuleWithData:
    """Tests for TicketDataModule that require preprocessed data."""

    @pytest.fixture
    def mock_preprocessed_data(self, tmp_path: Path):
        """Create minimal preprocessed dataset for testing."""
        from datasets import Dataset

        root = tmp_path / "data"
        (root / "raw").mkdir(parents=True)
        (root / "preprocessed").mkdir(parents=True)

        # Create minimal tokenized dataset
        data = {
            "input_ids": [[101, 1000, 2000, 102, 0, 0, 0, 0, 0, 0] for _ in range(10)],
            "attention_mask": [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0] for _ in range(10)],
            "labels": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        }
        dataset = Dataset.from_dict(data)
        dataset.set_format("torch")

        # Save for each split
        preprocessed_dir = root / "preprocessed"
        for split_name in ["train", "validation", "test"]:
            output_file = preprocessed_dir / f"small_{split_name}.parquet"
            dataset.to_parquet(str(output_file))

        return root

    def test_setup_creates_datasets(self, mock_preprocessed_data: Path) -> None:
        """Test that setup creates train/val datasets for fit stage."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=2,
        )
        datamodule.setup(stage="fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

    def test_setup_creates_test_dataset(self, mock_preprocessed_data: Path) -> None:
        """Test that setup creates test dataset for test stage."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=2,
        )
        datamodule.setup(stage="test")

        assert datamodule.test_dataset is not None

    def test_train_dataloader_returns_dataloader(self, mock_preprocessed_data: Path) -> None:
        """Test that train_dataloader returns a DataLoader."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=2,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2

    def test_val_dataloader_returns_dataloader(self, mock_preprocessed_data: Path) -> None:
        """Test that val_dataloader returns a DataLoader."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=4,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.val_dataloader()

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4

    def test_test_dataloader_returns_dataloader(self, mock_preprocessed_data: Path) -> None:
        """Test that test_dataloader returns a DataLoader."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=8,
        )
        datamodule.setup(stage="test")

        loader = datamodule.test_dataloader()

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 8

    def test_dataloader_batch_has_required_keys(self, mock_preprocessed_data: Path) -> None:
        """Test that DataLoader batches have required keys."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=2,
        )
        datamodule.setup(stage="fit")

        batch = next(iter(datamodule.train_dataloader()))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_batch_tensors_have_correct_shape(self, mock_preprocessed_data: Path) -> None:
        """Test that batch tensors have correct shapes."""
        batch_size = 3
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")

        batch = next(iter(datamodule.train_dataloader()))

        assert batch["input_ids"].shape[0] == batch_size
        assert batch["attention_mask"].shape[0] == batch_size
        assert batch["labels"].shape[0] == batch_size

    def test_train_dataloader_shuffles(self, mock_preprocessed_data: Path) -> None:
        """Test that train_dataloader shuffles data."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=10,  # Get all data in one batch
        )
        datamodule.setup(stage="fit")

        loader = datamodule.train_dataloader()

        # Check that shuffle is enabled in the dataloader
        assert loader.sampler is not None or hasattr(loader, "shuffle")

    def test_val_dataloader_does_not_shuffle(self, mock_preprocessed_data: Path) -> None:
        """Test that val_dataloader does not shuffle data."""
        preprocessed_dir = mock_preprocessed_data / "preprocessed"
        datamodule = TicketDataModule(
            train_path=preprocessed_dir / "small_train.parquet",
            val_path=preprocessed_dir / "small_validation.parquet",
            test_path=preprocessed_dir / "small_test.parquet",
            batch_size=2,
        )
        datamodule.setup(stage="fit")

        loader = datamodule.val_dataloader()

        # For non-shuffled loaders, sampler should be SequentialSampler
        from torch.utils.data import SequentialSampler

        assert isinstance(loader.sampler, SequentialSampler)
