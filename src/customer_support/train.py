from pathlib import Path
import logging

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback


from customer_support.datamodule import TicketDataModule
from customer_support.model import TicketClassificationModule

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # ---- sanity ----
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ---- reproducibility ----
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # ---- data ----
    datamodule = TicketDataModule(
        root=cfg.data.root,
        dataset_type=cfg.data.dataset_type,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        download=False,
    )

    # ---- model ----
    model = TicketClassificationModule(
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    # ---- callbacks ----
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=cfg.callbacks.patience,
        )
    ]

    if cfg.logging.save_model:
        output_dir = Path(cfg.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        callbacks: list[Callback] = []


        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f"model_{cfg.data.dataset_type}",
                monitor="val_accuracy",
                mode="max",
                save_top_k=1,
            )
        )

    # ---- logger ----
    logger = CSVLogger(
        save_dir=cfg.logging.log_dir,
        name="customer_support",
    )

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
    )

    # ---- train ----
    trainer.fit(model=model, datamodule=datamodule)

    # ---- metrics (fail loudly if missing) ----
    metrics = trainer.callback_metrics
    for key in ("val_accuracy", "val_loss"):
        if key not in metrics:
            raise RuntimeError(f"Expected metric '{key}' not logged")

    log.info(
        "Final metrics | val_acc=%.4f val_loss=%.4f",
        float(metrics["val_accuracy"]),
        float(metrics["val_loss"]),
    )


if __name__ == "__main__":
    train()
