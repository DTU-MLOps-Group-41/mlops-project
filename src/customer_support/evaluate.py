"""Model evaluation script for customer support ticket classification."""

from pathlib import Path

import torch
import typer
from loguru import logger

import lightning.pytorch as pl

from customer_support.datamodule import TicketDataModule
from customer_support.model import TicketClassificationModule


def evaluate(
    checkpoint_path: str | Path,
    data_root: str | Path = "data",
    dataset_type: str = "small",
    batch_size: int = 32,
    accelerator: str = "auto",
    devices: str | int = "auto",
    num_workers: int = 0,
) -> dict[str, float]:
    """Evaluate the customer support ticket classifier on test/validation datasets.

    Args:
        checkpoint_path: Path to the model checkpoint file.
        data_root: Root directory for dataset files.
        dataset_type: Dataset size - "small", "medium", or "full".
        batch_size: Evaluation batch size.
        accelerator: Lightning accelerator ("auto", "cpu", "gpu").
        devices: Number of devices or "auto".
        num_workers: Number of DataLoader workers.

    Returns:
        Dictionary with evaluation metrics (test_loss, test_accuracy, val_loss, val_accuracy).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        msg = f"Checkpoint file not found: {checkpoint_file}"
        raise FileNotFoundError(msg)

    # Set matmul precision for better performance on supported hardware
    torch.set_float32_matmul_precision("medium")

    logger.info(f"{'=' * 60}")
    logger.info("Model Evaluation Configuration:")
    logger.info(f"  Checkpoint: {checkpoint_file}")
    logger.info(f"  Dataset: {dataset_type} (root: {data_root})")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Accelerator: {accelerator}")
    logger.info(f"  Devices: {devices}")
    logger.info(f"{'=' * 60}")

    # Initialize DataModule
    datamodule = TicketDataModule(
        root=data_root,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        download=False,
    )

    # Load model from checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_file}")
    model = TicketClassificationModule.load_from_checkpoint(checkpoint_file)
    model.eval()

    # Initialize Trainer for evaluation
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_progress_bar=True,
    )

    logger.info("Evaluating on test dataset...")
    test_results = trainer.test(model=model, datamodule=datamodule, verbose=True)

    # Extract metrics from results
    test_metrics = test_results[0] if test_results else {}

    # Format metrics
    evaluation_metrics = {
        "test_loss": float(test_metrics.get("test_loss", 0.0)),
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
    }

    logger.info(f"{'=' * 60}")
    logger.info("Evaluation Complete - Results:")
    logger.info(f"  Test Loss: {evaluation_metrics['test_loss']:.4f}")
    logger.info(f"  Test Accuracy: {evaluation_metrics['test_accuracy']:.4f}")
    logger.info(f"{'=' * 60}")

    return evaluation_metrics


app = typer.Typer(help="Customer support model evaluation")


@app.command(name="evaluate")
def evaluate_command(
    checkpoint: str = typer.Argument(
        ...,
        help="Path to the model checkpoint file",
    ),
    dataset_type: str = typer.Option("small", "-d", "--dataset-type", help="Dataset size: small, medium, or full"),
    batch_size: int = typer.Option(32, "-b", "--batch-size", help="Evaluation batch size"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Lightning accelerator (auto, cpu, gpu, tpu)"),
    devices: str = typer.Option("auto", "--devices", help="Number of devices or 'auto'"),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers"),
) -> None:
    """Evaluate the customer support ticket priority classifier.

    Examples:
        uv run src/customer_support/evaluate.py models/model_small.ckpt
        uv run src/customer_support/evaluate.py models/model_small.ckpt -d medium
        uv run src/customer_support/evaluate.py models/model_small.ckpt -b 16 --accelerator gpu
    """
    # Parse devices if it's a string number
    parsed_devices: str | int = devices
    if devices != "auto":
        try:
            parsed_devices = int(devices)
        except ValueError:
            parsed_devices = devices

    evaluate(
        checkpoint_path=checkpoint,
        data_root="data",
        dataset_type=dataset_type,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=parsed_devices,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    app()
