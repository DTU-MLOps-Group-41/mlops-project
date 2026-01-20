"""PyTorch Lightning Module for customer support ticket classification."""

import torch
import lightning.pytorch as pl
from transformers import DistilBertForSequenceClassification
from torchmetrics.classification import MulticlassAccuracy
from hydra.utils import instantiate
from omegaconf import DictConfig

from customer_support.data import LABEL_MAP


class TicketClassificationModule(pl.LightningModule):
    """Lightning Module for customer support ticket priority classification.

    Wraps DistilBertForSequenceClassification with PyTorch Lightning integration,
    providing automatic training/validation/test steps with metric logging.

    Args:
        model_name: HuggingFace model identifier (default: "distilbert-base-multilingual-cased")
        num_classes: Number of output classes (default: calculated from LABEL_MAP)
        learning_rate: Learning rate for AdamW optimizer (default: 5e-5)
        weight_decay: Weight decay for AdamW optimizer (default: 0.01)

    Example:
        >>> model = TicketClassificationModule(learning_rate=5e-5)
        >>> trainer = pl.Trainer(max_epochs=3)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        num_classes: int | None = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        lr_scheduler_config: DictConfig | None = None,
    ) -> None:
        super().__init__()

        # Calculate num_classes from LABEL_MAP if not provided
        if num_classes is None:
            num_classes = len(set(LABEL_MAP.values()))

        self.save_hyperparameters(ignore=["lr_scheduler_config"])

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_config = lr_scheduler_config

        # Initialize the transformer model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
        )
        # Set to train mode to avoid Lightning warning about modules in eval mode
        self.model.train()

        # Initialize metrics using torchmetrics (separate instances for train/val/test)
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            labels: Optional labels for computing loss

        Returns:
            SequenceClassifierOutput with loss (if labels provided) and logits
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - compute loss and log metrics.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels
            batch_idx: Index of the current batch

        Returns:
            Loss tensor for backpropagation
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Update and log metrics (sync_dist=True for distributed training)
        self.train_accuracy(predictions, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step - compute loss and accuracy.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels
            batch_idx: Index of the current batch
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Update and log metrics (sync_dist=True for distributed training)
        self.val_accuracy(predictions, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step - compute accuracy.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels
            batch_idx: Index of the current batch
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Update and log metrics (sync_dist=True for distributed training)
        self.test_accuracy(predictions, batch["labels"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Predict step - return predictions and labels for visualization.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels
            batch_idx: Index of the current batch

        Returns:
            Dictionary with predictions and labels tensors
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        predictions = torch.argmax(outputs.logits, dim=-1)
        return {"predictions": predictions, "labels": batch["labels"]}

    def configure_optimizers(self):
        """Configure AdamW optimizer with optional LR scheduler via Hydra instantiate.

        Returns:
            AdamW optimizer, or dict with optimizer and lr_scheduler if scheduler is configured
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Check if scheduler is configured (has _target_)
        if self.lr_scheduler_config is None or self.lr_scheduler_config.get("_target_") is None:
            return optimizer

        # Use Hydra instantiate with runtime parameters
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = instantiate(
            self.lr_scheduler_config,
            optimizer=optimizer,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    model = TicketClassificationModule()
    print(model.model)
    for name, param in model.model.named_parameters():
        print(f"{name}: {param.size()}")

    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Total number of parameters: {total_params}")
