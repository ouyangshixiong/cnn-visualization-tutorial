"""CIFAR-10 DataModule for CNN visualization using PyTorch Lightning."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR-10 DataModule with visualization support."""
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def prepare_data(self):
        """Download CIFAR-10 dataset."""
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        """Setup transforms and datasets."""
        transform_list = [transforms.ToTensor()]
        if self.hparams.normalize:
            transform_list.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        
        transform = transforms.Compose(transform_list)
        
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                self.hparams.data_dir, train=True, transform=transform
            )
            self.val_dataset = datasets.CIFAR10(
                self.hparams.data_dir, train=False, transform=transform
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.hparams.data_dir, train=False, transform=transform
            )
    
    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )
    
    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )