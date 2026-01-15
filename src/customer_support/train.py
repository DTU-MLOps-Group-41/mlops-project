from pathlib import Path

import torch
import torch.optim as optim
import typer
from loguru import logger
from torch.utils.data import DataLoader

from customer_support.data import TicketDataset
from customer_support.model import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    data_root: str | Path = "data",
    dataset_type: str = "small",
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    patience: int = 2,
    output_dir: str | Path = "models",
    save_model: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    """Train the customer support ticket classifier.

    Args:
        data_root: Root directory for dataset files.
        dataset_type: Dataset size - "small", "medium", or "full".
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW optimizer.
        num_epochs: Maximum number of training epochs.
        weight_decay: Weight decay for AdamW optimizer.
        patience: Early stopping patience (epochs without improvement).
        output_dir: Directory to save model checkpoints.
        save_model: Whether to save the best model checkpoint.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with final training metrics (train_loss, val_loss, train_accuracy, val_accuracy).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"{'=' * 60}")
    logger.info("Training Configuration:")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Dataset: {dataset_type} (root: {data_root})")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Max epochs: {num_epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"{'=' * 60}")

    train_dataset = TicketDataset(root=data_root, split="train", dataset_type=dataset_type)
    val_dataset = TicketDataset(root=data_root, split="validation", dataset_type=dataset_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    model = get_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    final_metrics = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "train_accuracy": 0.0,
        "val_accuracy": 0.0,
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        logger.debug(f"Starting epoch {epoch + 1}/{num_epochs}")
        logger.debug(f"Number of batches: {len(train_loader)}")
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            logger.debug(f"  Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
            final_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
            logger.info(f"  New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if save_model and best_model_state is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / f"model_{dataset_type}.pt"
        torch.save(best_model_state, checkpoint_path)
        logger.success(f"Model saved to: {checkpoint_path}")

    logger.info(f"{'=' * 60}")
    logger.info("Training Complete - Final Metrics:")
    logger.info(f"  Train Loss: {final_metrics['train_loss']:.4f}")
    logger.info(f"  Train Accuracy: {final_metrics['train_accuracy']:.4f}")
    logger.info(f"  Val Loss: {final_metrics['val_loss']:.4f}")
    logger.info(f"  Val Accuracy: {final_metrics['val_accuracy']:.4f}")
    logger.info(f"{'=' * 60}")

    return final_metrics


app = typer.Typer(help="Customer support model training")


@app.command(name="train")
def train_command(
    dataset_type: str = typer.Option("small", "-d", "--dataset-type", help="Dataset size: small, medium, or full"),
    batch_size: int = typer.Option(32, "-b", "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", "--learning-rate", help="Learning rate for optimizer"),
    num_epochs: int = typer.Option(3, "-e", "--epochs", help="Maximum number of training epochs"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay for optimizer"),
    patience: int = typer.Option(2, "-p", "--patience", help="Early stopping patience"),
    output_dir: str = typer.Option("models", "-o", "--output-dir", help="Directory to save model checkpoints"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save model checkpoint"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Train the customer support ticket priority classifier.

    Examples:
        uv run src/customer_support/train.py train
        uv run src/customer_support/train.py train -d small -b 16 --lr 3e-5 -e 5
        uv run src/customer_support/train.py train --dataset-type medium --epochs 10
    """
    train(
        data_root="data",
        dataset_type=dataset_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        patience=patience,
        output_dir=output_dir,
        save_model=not no_save,
        seed=seed,
    )


if __name__ == "__main__":
    app()
