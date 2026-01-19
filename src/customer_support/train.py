"""Training script for customer support ticket classifier using PyTorch Lightning."""

from pathlib import Path

import torch
import typer
from loguru import logger


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from customer_support.datamodule import TicketDataModule
from customer_support.model import TicketClassificationModule

import hydra
from omegaconf import DictConfig


hydra.main(version_base=None, config_path="configs", config_name="default_config.yaml")
def train(cfg: DictConfig) -> None:
    # ------------------
    # Reproducibility
    # ------------------
    torch.manual_seed(cfg.seed)

    # ------------------
    # Data
    # ------------------
    data_root: Path = Path(cfg.data.root)
    dataset_type: str = cfg.data.dataset_type
    batch_size: int = cfg.data.batch_size
    num_workers: int = cfg.data.num_workers

    # ------------------
    # Model / Optimizer
    # ------------------
    learning_rate: float = cfg.model.learning_rate
    weight_decay: float = cfg.model.weight_decay

    # ------------------
    # Trainer
    # ------------------
    max_epochs: int = cfg.trainer.max_epochs
    accelerator = cfg.trainer.accelerator
    devices = cfg.trainer.devices
    precision = cfg.trainer.precision
    deterministic = cfg.trainer.deterministic
    log_every_n_steps = cfg.trainer.log_every_n_steps

    # ------------------
    # Callbacks
    # ------------------
    patience: int = cfg.callbacks.patience

    # ------------------
    # Logging / Outputs
    # ------------------
    log_dir: Path = Path(cfg.logging.log_dir)
    output_dir: Path = Path(cfg.logging.output_dir)
    save_model: bool = cfg.logging.save_model

    """Train the customer support ticket classifier using PyTorch Lightning.

    Args:
        data_root: Root directory for dataset files.
        dataset_type: Dataset size - "small", "medium", or "full".
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW optimizer.
        num_epochs: Maximum number of training epochs.
        weight_decay: Weight decay for AdamW optimizer.
        patience: Early stopping patience (epochs without improvement).
        output_dir: Directory to save model checkpoints.
        log_dir: Directory to save training logs.
        save_model: Whether to save the best model checkpoint.
        seed: Random seed for reproducibility.
        accelerator: Lightning accelerator ("auto", "cpu", "gpu", "tpu").
        devices: Number of devices or "auto".
        precision: Training precision ("16-mixed", "bf16-mixed", "32", etc.).
        num_workers: Number of DataLoader workers.

    Returns:
        Dictionary with final training metrics (train_loss, val_loss, train_accuracy, val_accuracy).
    """
    # Set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # Set matmul precision for better performance on supported hardware
    torch.set_float32_matmul_precision("medium")

    logger.info(f"{'=' * 60}")
    logger.info("Training Configuration (PyTorch Lightning):")
    logger.info(f"  Dataset: {dataset_type} (root: {data_root})")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Max epochs: {num_epochs}")
    logger.info(f"  Early stopping patience: {patience}")
    logger.info(f"  Accelerator: {accelerator}")
    logger.info(f"  Devices: {devices}")
    logger.info(f"  Precision: {precision}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"{'=' * 60}")

    # Initialize DataModule
    datamodule = TicketDataModule(
        root=data_root,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        download=False,
    )

    # Initialize Model
    model = TicketClassificationModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Configure callbacks
    callbacks = []

    # Early stopping callback - monitors val_accuracy (important for detecting catastrophic forgetting)
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=patience,
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Model checkpoint callback - saves best model by val_accuracy
    checkpoint_callback = None
    if save_model:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_path,
            filename=f"model_{dataset_type}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure CSV logger
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="customer_support",
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=csv_logger,
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Collect final metrics
    final_metrics = {
        "train_loss": float(trainer.callback_metrics.get("train_loss_epoch", 0.0)),
        "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
        "train_accuracy": float(trainer.callback_metrics.get("train_accuracy", 0.0)),
        "val_accuracy": float(trainer.callback_metrics.get("val_accuracy", 0.0)),
    }

    logger.info(f"{'=' * 60}")
    logger.info("Training Complete - Final Metrics:")
    logger.info(f"  Train Loss: {final_metrics['train_loss']:.4f}")
    logger.info(f"  Train Accuracy: {final_metrics['train_accuracy']:.4f}")
    logger.info(f"  Val Loss: {final_metrics['val_loss']:.4f}")
    logger.info(f"  Val Accuracy: {final_metrics['val_accuracy']:.4f}")
    logger.info(f"{'=' * 60}")

    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        logger.success(f"Best model saved to: {checkpoint_callback.best_model_path}")

    return final_metrics


# TODO: Add Hydra config support for hyperparameter management
# TODO: Add Weights & Biases logger integration


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
    log_dir: str = typer.Option("logs", "-l", "--log-dir", help="Directory to save training logs"),
    no_save: bool = typer.Option(False, "--no-save", help="Do not save model checkpoint"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Lightning accelerator (auto, cpu, gpu, tpu)"),
    devices: str = typer.Option("auto", "--devices", help="Number of devices or 'auto'"),
    precision: str | None = typer.Option(None, "--precision", help="Training precision (16-mixed, bf16-mixed, 32)"),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers"),
) -> None:
    """Train the customer support ticket priority classifier with PyTorch Lightning.

    Examples:
        uv run src/customer_support/train.py train
        uv run src/customer_support/train.py train -d small -b 16 --lr 3e-5 -e 5
        uv run src/customer_support/train.py train --dataset-type medium --epochs 10
        uv run src/customer_support/train.py train --accelerator gpu --precision 16-mixed
    """
    # Parse devices if it's a string number
    parsed_devices: str | int = devices
    if devices != "auto":
        try:
            parsed_devices = int(devices)
        except ValueError:
            parsed_devices = devices

    train(
        data_root="data",
        dataset_type=dataset_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        patience=patience,
        output_dir=output_dir,
        log_dir=log_dir,
        save_model=not no_save,
        seed=seed,
        accelerator=accelerator,
        devices=parsed_devices,
        precision=precision,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    app()
