"""
测试PyTorch可视化工具
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from visualization.pytorch.layer_visualizer import CNNLayerVisualizer


class TestCNN(nn.Module):
    """测试用的简单CNN"""
    def __init__(self):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 56 * 56, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class TestCNNLayerVisualizer:
    """测试CNN层可视化器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.model = TestCNN()
        self.visualizer = CNNLayerVisualizer(self.model)
        self.test_input = torch.randn(1, 3, 224, 224)
    
    def test_initialization(self):
        """测试初始化"""
        assert isinstance(self.visualizer, CNNLayerVisualizer)
        assert self.visualizer.model == self.model
    
    def test_get_layer_names(self):
        """测试获取层名称"""
        layer_names = self.visualizer.get_layer_names()
        assert 'conv1' in layer_names
        assert 'conv2' in layer_names
        assert 'pool' in layer_names
    
    def test_visualize_conv_filters(self):
        """测试卷积滤波器可视化"""
        fig = self.visualizer.visualize_conv_filters('conv1', max_filters=4)
        assert fig is not None
        plt.close(fig)
    
    def test_visualize_feature_maps(self):
        """测试特征图可视化"""
        fig = self.visualizer.visualize_feature_maps(
            self.test_input, 'conv1', max_maps=4
        )
        assert fig is not None
        plt.close(fig)
    
    def test_visualize_gradients(self):
        """测试梯度可视化"""
        fig = self.visualizer.visualize_gradients(
            self.test_input, target_class=0, layer_name='conv1'
        )
        assert fig is not None
        plt.close(fig)
    
    def test_invalid_layer_name(self):
        """测试无效的层名称"""
        with pytest.raises(ValueError):
            self.visualizer.visualize_conv_filters('invalid_layer')


if __name__ == "__main__":
    pytest.main([__file__])