"""ResNet CNN model for visualization using PaddlePaddle high-level API."""

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet18, resnet50


class ResNetCNNClassifier(nn.Layer):
    """ResNet-based CNN for visualization tasks."""
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        model_name: str = "resnet18"
    ):
        super().__init__()
        
        if model_name == "resnet18":
            self.backbone = resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(512, num_classes)
        elif model_name == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)