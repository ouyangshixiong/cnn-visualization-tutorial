"""ResNet CNN model for visualization using PyTorch Lightning."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet50
from torchmetrics import Accuracy


class ResNetCNNClassifier(pl.LightningModule):
    """ResNet-based CNN for visualization tasks."""
    
    def __init__(
        self, 
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        pretrained: bool = True,
        model_name: str = "resnet18"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained backbone
        if model_name == "resnet18":
            self.backbone = resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(512, num_classes)
        elif model_name == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        return self.backbone(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}