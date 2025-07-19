"""PaddlePaddle high-level API models for CNN visualization."""

from .simple_cnn import SimpleCNNClassifier
from .resnet_cnn import ResNetCNNClassifier

__all__ = ["SimpleCNNClassifier", "ResNetCNNClassifier"]