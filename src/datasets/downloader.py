"""High-level dataset downloader for CNN visualization."""

from typing import Dict, Callable
import torchvision.datasets as datasets


class DatasetDownloader:
    """High-level dataset downloader using built-in APIs."""
    
    SUPPORTED_DATASETS = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "imagenet": datasets.ImageNet,
    }
    
    @classmethod
    def download(
        cls, 
        dataset_name: str, 
        data_dir: str = "./data",
        split: str = "train"
    ) -> None:
        """Download dataset using high-level APIs."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from: {list(cls.SUPPORTED_DATASETS.keys())}"
            )
        
        dataset_class = cls.SUPPORTED_DATASETS[dataset_name]
        
        if dataset_name == "imagenet":
            dataset_class(data_dir, split=split, download=True)
        else:
            dataset_class(data_dir, train=True, download=True)
            dataset_class(data_dir, train=False, download=True)
    
    @classmethod
    def list_datasets(cls) -> Dict[str, str]:
        """List supported datasets with descriptions."""
        return {
            "cifar10": "CIFAR-10 (32x32 color images, 10 classes)",
            "cifar100": "CIFAR-100 (32x32 color images, 100 classes)",
            "mnist": "MNIST (28x28 grayscale digits)",
            "fashion_mnist": "Fashion-MNIST (28x28 fashion items)",
            "imagenet": "ImageNet (large-scale image classification)",
        }