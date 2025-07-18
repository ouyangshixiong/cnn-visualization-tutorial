"""
PyTorch Grad-CAM类激活图可视化
实现Grad-CAM、Grad-CAM++和Score-CAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional
import plotly.graph_objects as go
from PIL import Image


class GradCAM:
    """Grad-CAM实现"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """生成CAM热力图"""
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax().item()
        
        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize_cam(self, input_tensor: torch.Tensor, 
                     target_class: int = None, 
                     original_image: np.ndarray = None) -> plt.Figure:
        """可视化CAM"""
        cam = self.generate_cam(input_tensor, target_class)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图片
        if original_image is not None:
            axes[0].imshow(original_image)
        else:
            # 如果没有原始图片，显示输入张量
            img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM热力图
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加图
        if original_image is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 调整heatmap大小匹配原图
            heatmap = cv2.resize(heatmap, 
                               (original_image.shape[1], original_image.shape[0]))
            
            overlay = heatmap * 0.4 + original_image * 0.6
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            axes[2].imshow(overlay)
        else:
            img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[2].imshow(img)
            axes[2].imshow(cam, cmap='jet', alpha=0.5)
        
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'Grad-CAM Visualization - Class {target_class}')
        plt.tight_layout()
        return fig


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++实现"""
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """生成Grad-CAM++热力图"""
        self.model.eval()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++权重计算
        gradients_pow_2 = gradients ** 2
        gradients_pow_3 = gradients_pow_2 * gradients
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        
        alpha_num = gradients_pow_2
        alpha_denom = 2 * gradients_pow_2 + sum_activations * gradients_pow_3
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, 
                                 torch.ones_like(alpha_denom))
        
        alpha = alpha_num / alpha_denom
        
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class ScoreCAM:
    """Score-CAM实现"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hook()
    
    def _register_hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """生成Score-CAM热力图"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax().item()
            
            original_score = output[0, target_class]
            
            # 获取激活图
            activations = self.activations
            batch_size, num_channels, h, w = activations.shape
            
            # 上采样激活图到输入大小
            upsampled_activations = F.interpolate(
                activations, size=input_tensor.shape[2:], 
                mode='bilinear', align_corners=False
            )
            
            # 归一化激活图
            upsampled_activations = (upsampled_activations - 
                                   upsampled_activations.min()) / \
                                  (upsampled_activations.max() - 
                                   upsampled_activations.min() + 1e-8)
            
            # 计算每个激活图的权重
            weights = []
            for i in range(num_channels):
                # 创建掩码
                mask = upsampled_activations[0, i]
                masked_input = input_tensor * mask.unsqueeze(0)
                
                # 计算分数
                masked_output = self.model(masked_input)
                score = masked_output[0, target_class]
                
                weights.append(score.item())
            
            weights = np.array(weights)
            weights = weights - np.min(weights)
            weights = weights / (np.max(weights) + 1e-8)
            
            # 计算CAM
            cam = np.sum(weights.reshape(-1, 1, 1) * 
                        upsampled_activations[0].numpy(), axis=0)
            
            # 归一化
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam


def compare_cam_methods(model: nn.Module, target_layer: nn.Module,
                       input_tensor: torch.Tensor, 
                       original_image: np.ndarray = None,
                       target_class: int = None) -> plt.Figure:
    """比较不同的CAM方法"""
    
    # 初始化不同的CAM方法
    gradcam = GradCAM(model, target_layer)
    gradcam_plusplus = GradCAMPlusPlus(model, target_layer)
    scorecam = ScoreCAM(model, target_layer)
    
    # 生成热力图
    cam_gradcam = gradcam.generate_cam(input_tensor, target_class)
    cam_gradcampp = gradcam_plusplus.generate_cam(input_tensor, target_class)
    cam_scorecam = scorecam.generate_cam(input_tensor, target_class)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图片
    if original_image is not None:
        axes[0, 0].imshow(original_image)
    else:
        img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grad-CAM
    im1 = axes[0, 1].imshow(cam_gradcam, cmap='jet')
    axes[0, 1].set_title('Grad-CAM')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Grad-CAM++
    im2 = axes[1, 0].imshow(cam_gradcampp, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Score-CAM
    im3 = axes[1, 1].imshow(cam_scorecam, cmap='jet')
    axes[1, 1].set_title('Score-CAM')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.suptitle('Comparison of CAM Methods')
    plt.tight_layout()
    return fig