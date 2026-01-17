"""Tests for TicketClassificationModule (PyTorch Lightning Module)."""

import torch

from customer_support.data import LABEL_MAP
from customer_support.model import TicketClassificationModule


class TestTicketClassificationModule:
    """Tests for TicketClassificationModule."""

    def test_init_creates_model(self) -> None:
        """Test that __init__ creates the model with correct configuration."""
        module = TicketClassificationModule()

        assert module.model is not None
        assert module.num_classes == len(set(LABEL_MAP.values()))
        assert module.learning_rate == 5e-5
        assert module.weight_decay == 0.01

    def test_init_with_custom_hyperparameters(self) -> None:
        """Test initialization with custom hyperparameters."""
        module = TicketClassificationModule(
            learning_rate=1e-4,
            weight_decay=0.1,
            num_classes=5,
        )

        assert module.learning_rate == 1e-4
        assert module.weight_decay == 0.1
        assert module.num_classes == 5

    def test_forward_returns_output_with_loss_and_logits(self) -> None:
        """Test that forward returns model output with loss and logits."""
        module = TicketClassificationModule()

        # Create dummy input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.tensor([0, 1])

        output = module(input_ids, attention_mask, labels)

        assert hasattr(output, "loss")
        assert hasattr(output, "logits")
        assert output.loss is not None
        assert output.logits.shape == (batch_size, module.num_classes)

    def test_forward_without_labels_returns_logits_only(self) -> None:
        """Test that forward without labels returns logits but no loss."""
        module = TicketClassificationModule()

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = module(input_ids, attention_mask)

        assert output.loss is None
        assert output.logits.shape == (batch_size, module.num_classes)

    def test_configure_optimizers_returns_adamw(self) -> None:
        """Test that configure_optimizers returns AdamW with correct parameters."""
        lr = 5e-5
        weight_decay = 0.01
        module = TicketClassificationModule(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

        optimizer = module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == lr
        assert optimizer.defaults["weight_decay"] == weight_decay

    def test_training_step_returns_loss_tensor(self) -> None:
        """Test that training_step returns a scalar loss tensor."""
        module = TicketClassificationModule()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }

        loss = module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor
        assert loss.requires_grad  # Should be differentiable

    def test_validation_step_runs_without_error(self) -> None:
        """Test that validation_step runs without error."""
        module = TicketClassificationModule()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }

        # Should not raise any exceptions
        module.validation_step(batch, batch_idx=0)

    def test_test_step_runs_without_error(self) -> None:
        """Test that test_step runs without error."""
        module = TicketClassificationModule()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }

        # Should not raise any exceptions
        module.test_step(batch, batch_idx=0)

    def test_hparams_saved(self) -> None:
        """Test that hyperparameters are saved for checkpointing."""
        module = TicketClassificationModule(
            learning_rate=1e-4,
            weight_decay=0.05,
            num_classes=3,
        )

        assert hasattr(module, "hparams")
        assert module.hparams["learning_rate"] == 1e-4
        assert module.hparams["weight_decay"] == 0.05
        assert module.hparams["num_classes"] == 3

    def test_metrics_initialized(self) -> None:
        """Test that torchmetrics are properly initialized."""
        module = TicketClassificationModule()

        assert module.train_accuracy is not None
        assert module.val_accuracy is not None
        assert module.test_accuracy is not None

    def test_predict_step_returns_predictions_and_labels(self) -> None:
        """Test that predict_step returns dict with predictions and labels."""
        module = TicketClassificationModule()

        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }

        result = module.predict_step(batch, batch_idx=0)

        assert "predictions" in result
        assert "labels" in result
        assert result["predictions"].shape == (2,)
        assert result["labels"].shape == (2,)
