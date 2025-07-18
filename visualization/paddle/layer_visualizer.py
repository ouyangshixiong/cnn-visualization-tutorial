"""
PaddlePaddle CNN层可视化工具
提供卷积层、池化层、激活层的完整可视化功能
"""

import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CNNLayerVisualizer:
    """PaddlePaddle CNN层可视化器"""
    
    def __init__(self, model: nn.Layer):
        self.model = model
        self.activation_hooks = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(name):
            def hook(layer, input, output):
                self.activation_hooks[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(layer, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, layer in self.model.named_sublayers():
            if isinstance(layer, (nn.Conv2D, nn.MaxPool2D, nn.AvgPool2D, nn.ReLU)):
                layer.register_forward_post_hook(forward_hook(name))
                layer.register_backward_post_hook(backward_hook(name))
    
    def visualize_conv_filters(self, layer_name: str, max_filters: int = 16) -> plt.Figure:
        """可视化卷积层滤波器权重"""
        layer = dict(self.model.named_sublayers())[layer_name]
        if not isinstance(layer, nn.Conv2D):
            raise ValueError(f"Layer {layer_name} is not a Conv2D layer")
        
        weights = layer.weight.numpy()
        num_filters = min(weights.shape[0], max_filters)
        
        n_cols = 4
        n_rows = (num_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(num_filters):
            filter_weights = weights[i]
            if filter_weights.shape[0] == 3:  # RGB
                filter_img = np.transpose(filter_weights, (1, 2, 0))
                filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
                axes[i].imshow(filter_img)
            else:  # 单通道或深度可分离
                filter_img = filter_weights[0]
                axes[i].imshow(filter_img, cmap='viridis')
            
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Convolutional Filters - {layer_name}')
        plt.tight_layout()
        return fig
    
    def visualize_feature_maps(self, input_tensor: paddle.Tensor, 
                             layer_name: str, max_maps: int = 16) -> plt.Figure:
        """可视化特征图"""
        self.model.eval()
        with paddle.no_grad():
            _ = self.model(input_tensor)
        
        if layer_name not in self.activation_hooks:
            raise ValueError(f"Layer {layer_name} not found or no activation")
        
        feature_maps = self.activation_hooks[layer_name].numpy()
        num_maps = min(feature_maps.shape[1], max_maps)
        
        n_cols = 4
        n_rows = (num_maps + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(num_maps):
            feature_map = feature_maps[0, i]
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps - {layer_name}')
        plt.tight_layout()
        return fig
    
    def visualize_feature_maps_interactive(self, input_tensor: paddle.Tensor, 
                                         layer_name: str) -> go.Figure:
        """交互式特征图可视化"""
        self.model.eval()
        with paddle.no_grad():
            _ = self.model(input_tensor)
        
        feature_maps = self.activation_hooks[layer_name].numpy()
        num_channels = feature_maps.shape[1]
        
        # 创建子图
        n_cols = 8
        n_rows = (num_channels + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'Ch {i}' for i in range(num_channels)],
            horizontal_spacing=0.05,
            vertical_spacing=0.05
        )
        
        for i in range(num_channels):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            heatmap = go.Heatmap(
                z=feature_maps[0, i],
                colorscale='Viridis',
                showscale=False
            )
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title=f'Interactive Feature Maps - {layer_name}',
            height=100 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def visualize_gradients(self, input_tensor: paddle.Tensor, 
                          target_class: int, layer_name: str) -> plt.Figure:
        """可视化梯度"""
        self.model.train()
        input_tensor.stop_gradient = False
        
        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward()
        
        if layer_name not in self.gradients:
            raise ValueError(f"No gradients for layer {layer_name}")
        
        gradients = self.gradients[layer_name].numpy()
        num_channels = gradients.shape[1]
        
        n_cols = 4
        n_rows = (num_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i in range(min(num_channels, 16)):
            grad_map = gradients[0, i]
            im = axes[i].imshow(grad_map, cmap='coolwarm', 
                              vmin=grad_map.min(), vmax=grad_map.max())
            axes[i].set_title(f'Gradient {i}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # 隐藏多余的子图
        for i in range(min(num_channels, 16), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Gradients - {layer_name}')
        plt.tight_layout()
        return fig
    
    def get_layer_names(self) -> List[str]:
        """获取所有可可视化的层名称"""
        return [name for name, layer in self.model.named_sublayers() 
                if isinstance(layer, (nn.Conv2D, nn.MaxPool2D, nn.AvgPool2D, nn.ReLU))]