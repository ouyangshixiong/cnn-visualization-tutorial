"""Model evaluation script for CNN visualization."""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Evaluate a trained model."""
    
    # Initialize model (will load from checkpoint if provided)
    if cfg.get("checkpoint_path"):
        model = pl.LightningModule.load_from_checkpoint(cfg.checkpoint_path)
    else:
        model = hydra.utils.instantiate(cfg.model)
    
    # Initialize data module
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Evaluate the model
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()