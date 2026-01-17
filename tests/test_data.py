"""Tests for customer support data module.

This test suite validates the data processing pipeline for customer support ticket classification.
Tests focus on data contracts, reproducibility, and correct handling of edge cases.
Following MLOps testing principles: tests are fast, deterministic, and CI-friendly.
"""

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from customer_support.data import LABEL_MAP, TicketDataset
from tests import _PATH_DATA

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
def sample_raw_dataframe() -> pd.DataFrame:
    """Small sample dataframe with all 5 priority levels for testing."""
    return pd.DataFrame(
        {
            "body": [
                "My laptop won't turn on",
                "Password reset needed",
                "Server is down urgently",
                "Need access to shared folder",
                "Critical system failure - production down",
                None,
            ],
            "priority": ["very_low", "low", "Medium", "HIGH", "critical", "low"],
            "other_column": [1, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def sample_clean_dataframe() -> pd.DataFrame:
    """Cleaned dataframe with all 5 priority levels for testing."""
    return pd.DataFrame(
        {
            "body": [
                "My laptop won't turn on",
                "Password reset needed",
                "Server is down urgently",
                "Need access to shared folder",
                "Critical system failure - production down",
            ],
            "priority": ["very_low", "low", "medium", "high", "critical"],
        }
    )


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

    # Save each split as a separate Parquet file (matching the implementation)
    preprocessed_dir = temp_data_root / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        output_file = preprocessed_dir / f"small_{split_name}.parquet"
        dataset_dict[split_name].to_parquet(output_file)

    return temp_data_root


# ============================================================================
# STATIC METHOD TESTS
# ============================================================================


def test_clean_dataframe(sample_raw_dataframe) -> None:
    """Test dataframe cleaning removes missing values and selects columns."""
    result = TicketDataset._clean_dataframe(sample_raw_dataframe)

    assert len(result) == 5, f"Expected 5 rows after cleaning (1 NaN removed), got {len(result)}"
    assert list(result.columns) == ["body", "priority"], (
        f"Expected ['body', 'priority'] columns, got {list(result.columns)}"
    )
    assert result["body"].dtype == pd.StringDtype(), f"Expected body to be StringDtype, got {result['body'].dtype}"
    assert result.isna().sum().sum() == 0, "Cleaned dataframe should not contain any NaN values"


def test_encode_labels(sample_clean_dataframe) -> None:
    """Test label encoding converts priority to integers."""
    result = TicketDataset._encode_labels(sample_clean_dataframe, LABEL_MAP)

    assert "labels" in result.columns, "Encoded dataframe must contain 'labels' column"
    priorities = ["very_low", "low", "medium", "high", "critical"]
    expected_labels = [LABEL_MAP[p] for p in priorities]
    assert result["labels"].tolist() == expected_labels, (
        f"Expected labels {expected_labels}, got {result['labels'].tolist()}"
    )
    assert result["labels"].dtype == int, f"Labels should be int type, got {result['labels'].dtype}"


def test_encode_labels_handles_unknown() -> None:
    """Test that unknown labels are dropped."""
    df = pd.DataFrame(
        {
            "body": ["test1", "test2", "test3"],
            "priority": ["low", "unknown", "invalid_priority"],
        }
    )

    result = TicketDataset._encode_labels(df)

    assert len(result) == 1, f"Expected 1 row after dropping unknown labels, got {len(result)}"
    assert result["labels"].tolist() == [LABEL_MAP["low"]], (
        f"Expected label [{LABEL_MAP['low']}] for 'low' priority, got {result['labels'].tolist()}"
    )


def test_tokenize_dataset() -> None:
    """Test tokenization adds expected columns."""
    df = pd.DataFrame(
        {
            "body": ["Short text", "Another ticket with more words"],
            "labels": [0, 1],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset)

    assert "input_ids" in result.column_names, "Tokenized dataset must contain 'input_ids'"
    assert "attention_mask" in result.column_names, "Tokenized dataset must contain 'attention_mask'"
    assert "labels" in result.column_names, "Tokenized dataset must preserve 'labels' column"
    assert "body" not in result.column_names, "Tokenized dataset should not contain 'body' column"
    assert len(result) == 2, f"Tokenization should preserve dataset length, got {len(result)} samples"


def test_split_dataset() -> None:
    """Test dataset splitting creates correct splits."""
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(100)],
            "labels": [i % 3 for i in range(100)],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._split_dataset(dataset, test_size=0.2, val_size=0.5, seed=42)

    assert isinstance(result, DatasetDict), f"Expected DatasetDict, got {type(result)}"
    assert set(result.keys()) == {"train", "validation", "test"}, (
        f"Expected train/val/test splits, got {set(result.keys())}"
    )
    assert len(result["train"]) == 80, f"Expected 80 training samples, got {len(result['train'])}"
    assert len(result["validation"]) == 10, f"Expected 10 validation samples, got {len(result['validation'])}"
    assert len(result["test"]) == 10, f"Expected 10 test samples, got {len(result['test'])}"


def test_split_dataset_reproducible() -> None:
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

    assert result1["train"]["labels"] == result2["train"]["labels"], "Same seed should produce identical train splits"
    assert result1["validation"]["labels"] == result2["validation"]["labels"], (
        "Same seed should produce identical validation splits"
    )
    assert result1["test"]["labels"] == result2["test"]["labels"], "Same seed should produce identical test splits"


# ============================================================================
# TICKETDATASET CLASS TESTS
# ============================================================================


class TestTicketDataset:
    """Tests for TicketDataset class."""

    def test_load_existing_data(self, mock_preprocessed_dataset) -> None:
        """Test loading preprocessed data."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small", download=False)

        assert isinstance(dataset, TorchDataset), (
            f"TicketDataset should be instance of torch Dataset, got {type(dataset)}"
        )
        assert len(dataset) > 0, "Dataset should contain at least one sample"

        sample = dataset[0]
        assert isinstance(sample, dict), f"Sample should be dict, got {type(sample)}"
        assert "input_ids" in sample, "Sample must contain 'input_ids'"
        assert "attention_mask" in sample, "Sample must contain 'attention_mask'"
        assert "labels" in sample, "Sample must contain 'labels'"

    def test_raises_if_not_found(self, temp_data_root) -> None:
        """Test that RuntimeError is raised if data not found and download=False."""
        with pytest.raises(RuntimeError, match="Dataset not found"):
            TicketDataset(root=temp_data_root, split="train", dataset_type="small", download=False)

    def test_invalid_split(self, mock_preprocessed_dataset) -> None:
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be one of"):
            TicketDataset(root=mock_preprocessed_dataset, split="invalid", dataset_type="small")

    def test_invalid_dataset_type(self, mock_preprocessed_dataset) -> None:
        """Test that invalid dataset_type raises ValueError."""
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="invalid")

    def test_with_transform(self, mock_preprocessed_dataset) -> None:
        """Test that transform is applied in __getitem__."""

        def add_flag(sample):
            sample["transformed"] = True
            return sample

        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small", transform=add_flag)

        sample = dataset[0]
        assert "transformed" in sample, "Transform should add 'transformed' flag to sample"
        assert sample["transformed"] is True, "Transform flag should be True"

    def test_with_target_transform(self, mock_preprocessed_dataset) -> None:
        """Test that target_transform is applied to labels."""

        def add_ten(label):
            return label + 10

        dataset = TicketDataset(
            root=mock_preprocessed_dataset, split="train", dataset_type="small", target_transform=add_ten
        )

        sample = dataset[0]
        # Original labels are 0, 1 from mock_preprocessed_dataset
        # After transform, should be >= 10
        assert sample["labels"] >= 10, f"Target transform should add 10 to labels, got {sample['labels']}"

    def test_repr(self, mock_preprocessed_dataset) -> None:
        """Test string representation."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        repr_str = repr(dataset)
        assert "TicketDataset" in repr_str, "Repr should contain class name 'TicketDataset'"
        assert "split=train" in repr_str, "Repr should contain split information"
        assert "dataset_type=small" in repr_str, "Repr should contain dataset_type information"

    def test_get_label_map(self) -> None:
        """Test class method returns label mapping."""
        label_map = TicketDataset.get_label_map()
        assert isinstance(label_map, dict), f"get_label_map should return dict, got {type(label_map)}"
        assert label_map == LABEL_MAP, f"Label map mismatch: {label_map}"
        assert len(label_map) == len(LABEL_MAP), (
            f"Label map should have {len(LABEL_MAP)} priority levels, got {len(label_map)}"
        )

    def test_len(self, mock_preprocessed_dataset) -> None:
        """Test __len__ returns correct size."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        assert len(dataset) == 2, f"Expected 2 training samples in mock dataset, got {len(dataset)}"

    def test_getitem(self, mock_preprocessed_dataset) -> None:
        """Test __getitem__ returns correct sample."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split="train", dataset_type="small")

        sample = dataset[0]
        assert isinstance(sample, dict), f"Sample should be dict, got {type(sample)}"
        assert all(key in sample for key in ["input_ids", "attention_mask", "labels"]), "Sample missing required keys"

    def test_check_exists_false(self, temp_data_root) -> None:
        """Test _check_exists returns False when data missing."""
        dataset = TicketDataset.__new__(TicketDataset)
        dataset.processed_dir = temp_data_root / "preprocessed"
        dataset.dataset_type = "small"
        dataset.split = "train"

        assert dataset._check_exists() is False, "_check_exists should return False when data missing"

    def test_check_exists_true(self, mock_preprocessed_dataset) -> None:
        """Test _check_exists returns True when data present."""
        dataset = TicketDataset.__new__(TicketDataset)
        dataset.processed_dir = mock_preprocessed_dataset / "preprocessed"
        dataset.dataset_type = "small"
        dataset.split = "train"

        assert dataset._check_exists() is True, "_check_exists should return True when data present"

    @pytest.mark.parametrize("split", ["train", "validation", "test"])
    def test_load_different_splits(self, mock_preprocessed_dataset, split) -> None:
        """Test dataset can load different splits (parametrized)."""
        dataset = TicketDataset(root=mock_preprocessed_dataset, split=split, dataset_type="small", download=False)

        assert dataset.split == split, f"Dataset split should be {split}, got {dataset.split}"
        assert len(dataset) > 0, f"Dataset split '{split}' should contain at least one sample"


# ============================================================================
# INTEGRATION TESTS WITH DATA DIRECTORY (SKIP IF NOT AVAILABLE)
# ============================================================================


@pytest.mark.skipif(not (_PATH_DATA / "preprocessed").exists(), reason="Data files not found (DVC not ready)")
def test_load_real_preprocessed_data() -> None:
    """Test loading real preprocessed data when available."""
    dataset = TicketDataset(root=_PATH_DATA, split="train", dataset_type="small", download=False)

    assert len(dataset) > 0, "Real dataset should contain samples"
    sample = dataset[0]
    assert all(key in sample for key in ["input_ids", "attention_mask", "labels"]), (
        "Real dataset sample missing required keys"
    )


# ============================================================================
# LENGTH FILTERING AND PADDING TESTS
# ============================================================================


def test_tokenize_dataset_with_percentile_trim() -> None:
    """Test tokenization with percentile filtering using trim mode."""
    df = pd.DataFrame(
        {
            "body": [
                "Short",
                "Medium length text here",
                "This is a much longer text that should be trimmed to the percentile threshold",
                "Another very long text that exceeds the threshold and needs trimming to fit",
            ],
            "labels": [0, 1, 2, 3],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset, length_percentile=0.5, length_handling="trim")

    assert len(result) == 4, "Trim mode should preserve all samples"
    lengths = [len(result[i]["input_ids"]) for i in range(len(result))]
    assert len(set(lengths)) == 1, "All sequences should have the same length after padding"


def test_tokenize_dataset_with_percentile_drop() -> None:
    """Test tokenization with percentile filtering using drop mode."""
    df = pd.DataFrame(
        {
            "body": [
                "Short",
                "Medium length text here",
                "This is a much longer text that should be dropped when exceeding percentile threshold",
                "Another very long text that also exceeds the threshold and should be removed",
            ],
            "labels": [0, 1, 2, 3],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset, length_percentile=0.5, length_handling="drop")

    assert len(result) <= 4, "Drop mode may remove samples"
    lengths = [len(result[i]["input_ids"]) for i in range(len(result))]
    assert len(set(lengths)) == 1, "All sequences should have the same length after padding"


def test_tokenize_dataset_applies_uniform_padding() -> None:
    """Test that tokenization applies uniform padding to all sequences."""
    df = pd.DataFrame(
        {
            "body": ["Short", "Medium text", "Longer text here with more words"],
            "labels": [0, 1, 2],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset)

    lengths = [len(result[i]["input_ids"]) for i in range(len(result))]
    assert len(set(lengths)) == 1, f"Expected all sequences to have same length, got {set(lengths)}"

    attention_lengths = [len(result[i]["attention_mask"]) for i in range(len(result))]
    assert len(set(attention_lengths)) == 1, "Attention masks should have uniform length"
    assert lengths[0] == attention_lengths[0], "Input IDs and attention mask should have same length"


def test_tokenize_dataset_no_percentile_backwards_compatible() -> None:
    """Test tokenization without percentile maintains backwards compatibility."""
    df = pd.DataFrame(
        {
            "body": ["Test ticket 1", "Test ticket 2"],
            "labels": [0, 1],
        }
    )
    dataset = Dataset.from_pandas(df)

    result = TicketDataset._tokenize_dataset(dataset)

    assert "input_ids" in result.column_names, "Should contain input_ids"
    assert "attention_mask" in result.column_names, "Should contain attention_mask"
    assert "labels" in result.column_names, "Should contain labels"
    assert len(result) == 2, "Should preserve all samples"


def test_invalid_length_handling() -> None:
    """Test that invalid length_handling raises ValueError."""
    with pytest.raises(ValueError, match="length_handling must be one of"):
        TicketDataset.__new__(TicketDataset)
        TicketDataset(
            root="data",
            split="train",
            dataset_type="small",
            length_handling="invalid",
        )
