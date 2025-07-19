"""Tests for Lightning DataModules."""

import pytest
import os
import tempfile
from src.datasets.datamodules import CIFAR10DataModule


class TestCIFAR10DataModule:
    """Tests for CIFAR10DataModule."""
    
    @pytest.fixture
    def datamodule(self):
        """Create a CIFAR10DataModule instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = CIFAR10DataModule(
                data_dir=temp_dir,
                batch_size=4,
                num_workers=0  # Single-threaded for testing
            )
            yield dm
    
    def test_datamodule_creation(self, datamodule):
        """Test datamodule creation."""
        assert datamodule is not None
        assert datamodule.hparams.batch_size == 4
    
    def test_prepare_data(self, datamodule):
        """Test data preparation."""
        datamodule.prepare_data()
        assert os.path.exists(datamodule.hparams.data_dir)
    
    def test_setup(self, datamodule):
        """Test setup of datasets."""
        datamodule.prepare_data()
        datamodule.setup("fit")
        datamodule.setup("test")
        
        assert hasattr(datamodule, 'train_dataset')
        assert hasattr(datamodule, 'val_dataset')
        assert hasattr(datamodule, 'test_dataset')
    
    def test_dataloaders(self, datamodule):
        """Test dataloaders creation."""
        datamodule.prepare_data()
        datamodule.setup("fit")
        datamodule.setup("test")
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch size
        batch = next(iter(train_loader))
        assert len(batch[0]) == 4  # batch_size
        assert batch[0].shape[1:] == (3, 32, 32)  # CIFAR-10 image shape
    
    def test_no_normalize(self):
        """Test datamodule without normalization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = CIFAR10DataModule(
                data_dir=temp_dir,
                batch_size=4,
                normalize=False,
                num_workers=0
            )
            dm.prepare_data()
            dm.setup("fit")
            
            assert dm is not None
            train_loader = dm.train_dataloader()
            batch = next(iter(train_loader))
            assert batch[0].shape[1:] == (3, 32, 32)