"""Training script for customer support ticket classifier using PyTorch Lightning."""

from pathlib import Path
from typing import Literal, Optional

import torch
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.profilers import SimpleProfiler
from customer_support.datamodule import TicketDataModule
from customer_support.model import TicketClassificationModule

import hydra
from omegaconf import DictConfig, OmegaConf

ACCELERATOR_TY = Literal["auto", "cpu", "gpu", "tpu"]


# TODO: Fix logging
@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    """Train the customer support ticket classifier using PyTorch Lightning."""
    logger.info("Resolved Hydra config:\n" + OmegaConf.to_yaml(cfg))

    seed: int = cfg.seed
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    # Directories
    log_dir: Path = Path(cfg.paths.output_dir)
    output_dir: Path = Path(cfg.paths.model_dir)

    model_name: str = cfg.model_name

    # Dataset paths from Hydra config
    train_path: str = cfg.dataset.train_path
    val_path: str = cfg.dataset.val_path
    test_path: str = cfg.dataset.test_path
    dataset_name: str = cfg.dataset.name
    num_classes: int = cfg.dataset.num_classes

    accelerator: ACCELERATOR_TY = cfg.accelerator
    devices: list[int] | str | int = cfg.devices
    num_workers: int = cfg.num_workers
    save_model: bool = cfg.training.save_best_only


    # Model Hyperparameters
    batch_size: int = cfg.training.batch_size
    deterministic: bool = cfg.training.deterministic
    learning_rate: float = cfg.training.learning_rate
    log_every_n_steps: int = cfg.training.log_every_n_steps
    num_epochs: int = cfg.training.num_epochs
    patience: int = cfg.training.patience
    precision: Optional[_PRECISION_INPUT] = cfg.training.precision
    weight_decay: float = cfg.training.weight_decay
    checkpoint_monitor: str = cfg.training.checkpoint_monitor
    checkpoint_mode: str = cfg.training.checkpoint_mode
    checkpoint_save_top_k: int = cfg.training.checkpoint_save_top_k
    checkpoint_verbose: bool = cfg.training.checkpoint_verbose
    gradient_clip_val: Optional[float] = cfg.training.gradient_clip_val
    accumulate_grad_batches: int = cfg.training.accumulate_grad_batches
    val_check_interval: float = cfg.training.val_check_interval
    check_val_every_n_epoch: int = cfg.training.check_val_every_n_epoch
    lr_scheduler_cfg = cfg.training.get("lr_scheduler", None)

    # Wandb Config
    project = cfg.wandb.project
    entity=cfg.wandb.entity
    mode=cfg.wandb.mode


    # Profiling
    profiler = SimpleProfiler(dirpath=cfg.paths.output_dir, filename="profile_report")

    # Set matmul precision for better performance on supported hardware
    torch.set_float32_matmul_precision("medium")

    # Initialize DataModule
    datamodule = TicketDataModule(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Initialize Model
    model = TicketClassificationModule(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_config=lr_scheduler_cfg,
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
            filename=f"model_{dataset_name}",
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=checkpoint_save_top_k,
            verbose=checkpoint_verbose,
        )
        callbacks.append(checkpoint_callback)

    # Configure CSV logger
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="customer_support",
    )

    # Configure Wandb logger
    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        mode=mode,
        name=f"{model_name}-{dataset_name}",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        log_model="all" if cfg.training.save_best_only else False,
        save_dir=log_dir,
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=[csv_logger, wandb_logger],
        deterministic=deterministic,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        profiler=profiler
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)


    if getattr(wandb_logger, "experiment", None):
        wandb_logger.experiment.finish()

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


# TODO: Add Weights & Biases logger integration

if __name__ == "__main__":
    train()
