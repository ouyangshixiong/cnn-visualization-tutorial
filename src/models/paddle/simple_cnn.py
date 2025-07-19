"""Simple CNN model for visualization using PaddlePaddle high-level API."""

import paddle
import paddle.nn as nn


class SimpleCNN(nn.Layer):
    """Simple CNN backbone for visualization."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2D(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = paddle.flatten(x, start_axis=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNNClassifier(nn.Layer):
    """Simple CNN classifier with high-level API."""
    
    def __init__(self, num_classes: int = 10, learning_rate: float = 1e-3):
        super().__init__()
        self.backbone = SimpleCNN(num_classes)
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.backbone(x)