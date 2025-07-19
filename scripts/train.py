"""Unified training script for CNN visualization using high-level APIs."""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra import initialize, compose


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Set random seed
    pl.seed_everything(cfg.seed)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model)
    
    # Initialize data module
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Train the model
    trainer.fit(model, datamodule)
    
    # Test the model
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()