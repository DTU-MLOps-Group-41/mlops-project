"""Visualization script for customer support ticket classification model."""

from pathlib import Path

import torch
import typer
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


from customer_support.datamodule import TicketDataModule
from customer_support.model import TicketClassificationModule
from customer_support.data import LABEL_MAP


# Reverse label map for display
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_LABELS = sorted(set(LABEL_MAP.values()))
CLASS_NAMES = [REVERSE_LABEL_MAP[i] for i in CLASS_LABELS]


def visualize(
    checkpoint_path: str | Path,
    data_root: str | Path = "data",
    dataset_type: str = "small",
    batch_size: int = 32,
    output_dir: str | Path = "reports/figures",
    accelerator: str = "auto",
    devices: str | int = "auto",
    num_workers: int = 0,
) -> None:
    """Generate confusion matrix visualization for model predictions on test dataset.

    Args:
        checkpoint_path: Path to the model checkpoint file.
        data_root: Root directory for dataset files.
        dataset_type: Dataset size - "small", "medium", or "full".
        batch_size: Evaluation batch size.
        output_dir: Directory to save visualizations.
        accelerator: Lightning accelerator ("auto", "cpu", "gpu", "tpu").
        devices: Number of devices or "auto".
        num_workers: Number of DataLoader workers.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        msg = f"Checkpoint file not found: {checkpoint_file}"
        raise FileNotFoundError(msg)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set matmul precision
    torch.set_float32_matmul_precision("medium")

    logger.info(f"{'=' * 60}")
    logger.info("Model Visualization Configuration:")
    logger.info(f"  Checkpoint: {checkpoint_file}")
    logger.info(f"  Dataset: {dataset_type}")
    logger.info(f"  Output directory: {output_path}")
    logger.info(f"{'=' * 60}")

    # Load model and datamodule
    logger.info(f"Loading checkpoint from: {checkpoint_file}")
    model = TicketClassificationModule.load_from_checkpoint(checkpoint_file)
    model.eval()

    datamodule = TicketDataModule(
        root=data_root,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        download=False,
    )

    # Collect predictions on test set
    logger.info("Collecting predictions on test dataset...")
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Compute confusion matrix
    logger.info("Computing confusion matrix...")
    cm = confusion_matrix(all_labels, all_predictions, labels=CLASS_LABELS)

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot confusion matrix as heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # Set labels and title
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {dataset_type.upper()} Dataset", fontsize=14, fontweight="bold")

    # Set ticks
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            text_color = "white" if count > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                str(int(count)),
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save figure
    output_file = output_path / f"confusion_matrix_{dataset_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.success(f"Confusion matrix saved to: {output_file}")

    plt.close()


app = typer.Typer(help="Customer support model visualization")


@app.command(name="visualize")
def visualize_command(
    checkpoint: str = typer.Argument(
        ...,
        help="Path to the model checkpoint file",
    ),
    dataset_type: str = typer.Option("small", "-d", "--dataset-type", help="Dataset size: small, medium, or full"),
    batch_size: int = typer.Option(32, "-b", "--batch-size", help="Evaluation batch size"),
    output_dir: str = typer.Option("reports/figures", "-o", "--output-dir", help="Output directory for visualizations"),
    accelerator: str = typer.Option("auto", "--accelerator", help="Lightning accelerator (auto, cpu, gpu)"),
    devices: str = typer.Option("auto", "--devices", help="Number of devices or 'auto'"),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers"),
) -> None:
    """Generate confusion matrix visualization for the ticket classifier.

    Examples:
        uv run src/customer_support/visualize.py models/model_small.ckpt
        uv run src/customer_support/visualize.py models/model_small.ckpt -d medium
        uv run src/customer_support/visualize.py models/model_small.ckpt -o reports/figures
    """
    # Parse devices
    parsed_devices: str | int = devices
    if devices != "auto":
        try:
            parsed_devices = int(devices)
        except ValueError:
            parsed_devices = devices

    visualize(
        checkpoint_path=checkpoint,
        data_root="data",
        dataset_type=dataset_type,
        batch_size=batch_size,
        output_dir=output_dir,
        accelerator=accelerator,
        devices=parsed_devices,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    app()
