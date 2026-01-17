"""Tests for evaluate.py module."""

import pytest
from pathlib import Path

from customer_support.evaluate import evaluate


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_raises_file_not_found_for_missing_checkpoint(self, tmp_path: Path) -> None:
        """Test that evaluate raises FileNotFoundError for missing checkpoint."""
        fake_checkpoint = tmp_path / "nonexistent.ckpt"

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            evaluate(checkpoint_path=fake_checkpoint)

    def test_evaluate_raises_file_not_found_with_correct_path(self, tmp_path: Path) -> None:
        """Test that error message contains the checkpoint path."""
        fake_checkpoint = tmp_path / "missing_model.ckpt"

        with pytest.raises(FileNotFoundError, match="missing_model.ckpt"):
            evaluate(checkpoint_path=fake_checkpoint)
