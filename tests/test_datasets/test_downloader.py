"""Tests for dataset downloader."""

import pytest
import os
import tempfile
from src.datasets.downloader import DatasetDownloader


class TestDatasetDownloader:
    """Tests for DatasetDownloader."""
    
    def test_supported_datasets(self):
        """Test supported datasets list."""
        datasets = DatasetDownloader.list_datasets()
        assert "cifar10" in datasets
        assert "mnist" in datasets
        assert "fashion_mnist" in datasets
    
    def test_download_cifar10(self):
        """Test CIFAR-10 download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            DatasetDownloader.download("cifar10", temp_dir)
            
            # Check if files were created
            assert os.path.exists(temp_dir)
            assert os.path.exists(os.path.join(temp_dir, "cifar-10-batches-py"))
    
    def test_download_mnist(self):
        """Test MNIST download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            DatasetDownloader.download("mnist", temp_dir)
            
            # Check if files were created
            assert os.path.exists(temp_dir)
            assert os.path.exists(os.path.join(temp_dir, "MNIST"))
    
    def test_invalid_dataset(self):
        """Test invalid dataset name raises error."""
        with pytest.raises(ValueError):
            DatasetDownloader.download("invalid_dataset", "./data")