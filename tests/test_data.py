import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from customer_support.data import LABEL_MAP, TicketDataset


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_data_root(tmp_path):
    """Temporary data directory for testing."""
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "preprocessed").mkdir(parents=True)
    return root


@pytest.fixture
def mock_raw_csv(temp_data_root):
    """Create mock raw CSV for preprocessing tests."""
    csv_path = temp_data_root / "raw" / "small.csv"
    df = pd.DataFrame(
        {
            "body": ["Ticket 1", "Ticket 2", "Ticket 3", "Ticket 4"],
            "priority": ["low", "Medium", "HIGH", "low"],
            "other_col": [1, 2, 3, 4],
        }
    )
    df.to_csv(csv_path, index=False)
    return temp_data_root


@pytest.fixture
def mock_preprocessed_dataset(temp_data_root):
    """Create minimal preprocessed dataset for testing."""
    df = pd.DataFrame(
        {
            "body": ["Test ticket 1", "Test ticket 2", "Test ticket 3"],
            "labels": [0, 1, 2],
        }
    )

    dataset = Dataset.from_pandas(df)

    tokenized = TicketDataset._tokenize_dataset(dataset)

    dataset_dict = DatasetDict(
        {
            "train": tokenized.select(range(2)),
            "validation": tokenized.select([0]),
            "test": tokenized.select([1]),
        }
    )

    output_path = temp_data_root / "preprocessed" / "small"
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(output_path)

    return temp_data_root


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Small sample dataframe for testing."""
    return pd.DataFrame(
        {
            "body": [
                "My laptop won't turn on",
                "Password reset needed",
                "Server is down urgently",
                None,
            ],
            "priority": ["low", "Medium", "HIGH", "low"],
            "other_column": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def sample_clean_dataframe() -> pd.DataFrame:
    """Cleaned dataframe for testing."""
    return pd.DataFrame(
        {
            "body": [
                "My laptop won't turn on",
                "Password reset needed",
                "Server is down urgently",
            ],
            "priority": ["low", "medium", "high"],
        }
    )


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


def test_clean_dataframe(sample_raw_dataframe):
    """Test dataframe cleaning removes missing values and selects columns."""
    result = TicketDataset._clean_dataframe(sample_raw_dataframe)

    assert len(result) == 3
    assert list(result.columns) == ["body", "priority"]
    assert result["body"].dtype == pd.StringDtype()
    assert result.isna().sum().sum() == 0


def test_encode_labels(sample_clean_dataframe):
    """Test label encoding converts priority to integers."""
    result = TicketDataset._encode_labels(sample_clean_dataframe, LABEL_MAP)

    assert "labels" in result.columns
    assert result["labels"].tolist() == [0, 1, 2]
    assert result["labels"].dtype == int


def test_encode_labels_handles_unknown():
    """Test that unknown labels are dropped."""
    df = pd.DataFrame(
        {
            "body": ["test1", "test2"],
            "priority": ["low", "unknown"],
        }
    )

    result = TicketDataset._encode_labels(df)

    assert len(result) == 1
    assert result["labels"].tolist() == [0]


def test_tokenize_dataset():
    """Test tokenization adds expected columns."""
    df = pd.DataFrame(
        {
            "body": ["Short text", "Another ticket with more words"],
            "labels": [0, 1],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset)

    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names
    assert "labels" in result.column_names
    assert "body" not in result.column_names
    assert len(result) == 2


def test_split_dataset():
    """Test dataset splitting creates correct splits."""
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(100)],
            "labels": [i % 3 for i in range(100)],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._split_dataset(dataset, test_size=0.2, val_size=0.5, seed=42)

    assert isinstance(result, DatasetDict)
    assert set(result.keys()) == {"train", "validation", "test"}
    assert len(result["train"]) == 80
    assert len(result["validation"]) == 10
    assert len(result["test"]) == 10


def test_split_dataset_reproducible():
    """Test that splitting with same seed gives same results."""
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(50)],
            "labels": [i % 3 for i in range(50)],
        }
    )
    dataset = Dataset.from_pandas(df)

    result1 = TicketDataset._split_dataset(dataset, seed=42)
    result2 = TicketDataset._split_dataset(dataset, seed=42)

    assert result1["train"]["labels"] == result2["train"]["labels"]


# ============================================================================
# TICKETDATASET CLASS TESTS
# ============================================================================


class TestTicketDataset:
    """Tests for TicketDataset class."""

    def test_load_existing_data(self, mock_preprocessed_dataset):
        """Test loading preprocessed data."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small", download=False)

        assert isinstance(dataset, TorchDataset)
        assert len(dataset) > 0

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

    def test_raises_if_not_found(self, temp_data_root):
        """Test that RuntimeError is raised if data not found and download=False."""
        with pytest.raises(RuntimeError, match="Dataset not found"):
            TicketDataset(root=temp_data_root, split="train", dataset_type="small", download=False)

    def test_invalid_split(self, mock_preprocessed_dataset):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            TicketDataset(root=mock_preprocessed_dataset, split="invalid", dataset_type="small")

    def test_invalid_dataset_type(self, mock_preprocessed_dataset):
        """Test that invalid dataset_type raises ValueError."""
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="invalid")

    def test_with_transform(self, mock_preprocessed_dataset):
        """Test that transform is applied in __getitem__."""

        def add_flag(sample):
            sample["transformed"] = True
            return sample

        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small", transform=add_flag)

        sample = dataset[0]
        assert "transformed" in sample
        assert sample["transformed"] is True

    def test_with_target_transform(self, mock_preprocessed_dataset):
        """Test that target_transform is applied to labels."""

        def add_ten(label):
            return label + 10

        dataset = TicketDataset(
            root=mock_preprocessed_dataset, split="train", dataset_type="small", target_transform=add_ten
        )

        sample = dataset[0]
        # Original labels are 0, 1, 2 from mock_preprocessed_dataset
        # After transform, should be >= 10
        assert sample["labels"] >= 10

    def test_repr(self, mock_preprocessed_dataset):
        """Test string representation."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        repr_str = repr(dataset)
        assert "TicketDataset" in repr_str
        assert "split=train" in repr_str
        assert "dataset_type=small" in repr_str

    def test_get_label_map(self):
        """Test class method returns label mapping."""
        label_map = TicketDataset.get_label_map()
        assert isinstance(label_map, dict)
        assert label_map == {"low": 0, "medium": 1, "high": 2}

    def test_len(self, mock_preprocessed_dataset):
        """Test __len__ returns correct size."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        assert len(dataset) == 2

    def test_getitem(self, mock_preprocessed_dataset):
        """Test __getitem__ returns correct sample."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert all(key in sample for key in ["input_ids", "attention_mask", "labels"])

    def test_check_exists_false(self, temp_data_root):
        """Test _check_exists returns False when data missing."""
        dataset = TicketDataset.__new__(TicketDataset)
        dataset.processed_dir = temp_data_root / "preprocessed"
        dataset.dataset_type = "small"
        dataset.split = "train"

        assert dataset._check_exists() is False

    def test_check_exists_true(self, mock_preprocessed_dataset):
        """Test _check_exists returns True when data present."""
        dataset = TicketDataset.__new__(TicketDataset)
        dataset.processed_dir = mock_preprocessed_dataset / "preprocessed"
        dataset.dataset_type = "small"
        dataset.split = "train"

        assert dataset._check_exists() is True
