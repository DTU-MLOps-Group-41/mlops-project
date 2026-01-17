"""Tests for visualize.py module."""

import pytest
from pathlib import Path

from customer_support.visualize import (
    visualize,
    REVERSE_LABEL_MAP,
    CLASS_LABELS,
    CLASS_NAMES,
)


class TestVisualize:
    """Tests for visualize function."""

    def test_visualize_raises_file_not_found_for_missing_checkpoint(self, tmp_path: Path) -> None:
        """Test that visualize raises FileNotFoundError for missing checkpoint."""
        fake_checkpoint = tmp_path / "nonexistent.ckpt"

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            visualize(checkpoint_path=fake_checkpoint)

    def test_visualize_raises_file_not_found_with_correct_path(self, tmp_path: Path) -> None:
        """Test that error message contains the checkpoint path."""
        fake_checkpoint = tmp_path / "missing_model.ckpt"

        with pytest.raises(FileNotFoundError, match="missing_model.ckpt"):
            visualize(checkpoint_path=fake_checkpoint)


class TestLabelMaps:
    """Tests for label map constants."""

    def test_reverse_label_map_maps_indices_to_labels(self) -> None:
        """Test that REVERSE_LABEL_MAP maps each class index to a valid label."""
        from customer_support.data import LABEL_MAP

        # Each index in REVERSE_LABEL_MAP should correspond to a valid label in LABEL_MAP
        for idx, label in REVERSE_LABEL_MAP.items():
            assert label in LABEL_MAP
            assert LABEL_MAP[label] == idx

    def test_class_labels_are_sorted(self) -> None:
        """Test that CLASS_LABELS are sorted."""
        assert CLASS_LABELS == sorted(CLASS_LABELS)

    def test_class_names_match_class_labels(self) -> None:
        """Test that CLASS_NAMES correspond to CLASS_LABELS."""
        assert len(CLASS_NAMES) == len(CLASS_LABELS)
        for i, name in enumerate(CLASS_NAMES):
            assert REVERSE_LABEL_MAP[CLASS_LABELS[i]] == name
