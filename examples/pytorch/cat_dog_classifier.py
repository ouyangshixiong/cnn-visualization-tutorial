#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch猫狗分类器 + 完整可视化
在venv_linux虚拟环境中运行

使用说明：
1. 先激活虚拟环境：source venv_linux/bin/activate
2. 运行：python examples/pytorch/cat_dog_classifier.py
"""

import os
import sys
import warnings

# 确保在虚拟环境中运行
if not sys.prefix.endswith('venv_linux'):
    warnings.warn("建议在venv_linux虚拟环境中运行此脚本")

# 添加项目路径到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# 导入可视化工具
from visualization.pytorch.layer_visualizer import CNNLayerVisualizer
from visualization.pytorch.cam_visualizer import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, compare_cam_methods
)

# 配置matplotlib支持中文
#plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleCNN(nn.Module):
    """简单的CNN模型用于猫狗分类"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 第一层：卷积 + 批归一化 + 激活 + 池化
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        
        # 第二层
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        
        # 第三层
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 56 -> 28
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class CatDogDataset(Dataset):
    """猫狗数据集 - 使用虚拟数据演示"""
    
    def __init__(self, root_dir, transform=None, train=True, num_samples=100):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        self.num_samples = num_samples
        
        # 创建数据集目录（如果不存在）
        self.root_dir.mkdir(exist_ok=True)
        (self.root_dir / 'cats').mkdir(exist_ok=True)
        (self.root_dir / 'dogs').mkdir(exist_ok=True)
        
        # 生成虚拟图片数据
        self.images = []
        self.labels = []
        self._generate_virtual_data()
    
    def _generate_virtual_data(self):
        """生成虚拟的猫狗图片数据"""
        for i in range(self.num_samples):
            if i % 2 == 0:
                # 猫图片特征：偏暖色调，圆形特征
                img = self._create_cat_image()
                label = 0  # 猫
            else:
                # 狗图片特征：偏冷色调，长形特征
                img = self._create_dog_image()
                label = 1  # 狗
            
            self.images.append(img)
            self.labels.append(label)
    
    def _create_cat_image(self):
        """创建模拟的猫图片"""
        # 创建基础图像
        img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        
        # 添加猫脸特征（圆形区域）
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        radius = np.random.randint(30, 50)
        
        y, x = np.ogrid[:224, :224]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        
        # 暖色调
        img[mask] = [255, 200, 150]  # 橙色
        
        # 添加耳朵
        ear1 = (center[0] - radius//2, center[1] - radius)
        ear2 = (center[0] + radius//2, center[1] - radius)
        
        for ear in [ear1, ear2]:
            ear_mask = (x - ear[0]) ** 2 + (y - ear[1]) ** 2 <= (radius//3) ** 2
            img[ear_mask] = [200, 150, 100]  # 棕色
        
        return img
    
    def _create_dog_image(self):
        """创建模拟的狗图片"""
        # 创建基础图像
        img = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        
        # 添加狗脸特征（椭圆形区域）
        center = (np.random.randint(80, 140), np.random.randint(80, 140))
        a, b = np.random.randint(40, 60), np.random.randint(30, 45)
        
        y, x = np.ogrid[:224, :224]
        mask = ((x - center[0]) / a) ** 2 + ((y - center[1]) / b) ** 2 <= 1
        
        # 冷色调
        img[mask] = [150, 200, 255]  # 蓝色
        
        # 添加耳朵（长形）
        ear_y = center[1] - b
        ear_height = b//2
        ear_width = a//3
        
        ear_mask = (
            (abs(x - center[0]) <= ear_width) & 
            (abs(y - ear_y) <= ear_height)
        )
        img[ear_mask] = [100, 150, 200]  # 深蓝色
        
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # 转换为PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, label


def create_transforms():
    """创建数据转换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # 计算指标
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(100 * train_correct / train_total)
        val_accuracies.append(100 * val_correct / val_total)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('训练与验证损失')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue')
    ax2.plot(val_accuracies, label='验证准确率', color='red')
    ax2.set_title('训练与验证准确率')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习曲线对比
    ax3.plot(train_losses, label='训练损失', color='blue')
    ax3.plot(val_losses, label='验证损失', color='red', linestyle='--')
    ax3.set_title('损失对比')
    ax3.set_xlabel('轮次')
    ax3.set_ylabel('损失')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 准确率对比
    ax4.plot(train_accuracies, label='训练准确率', color='blue')
    ax4.plot(val_accuracies, label='验证准确率', color='red', linestyle='--')
    ax4.set_title('准确率对比')
    ax4.set_xlabel('轮次')
    ax4.set_ylabel('准确率 (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_pytorch.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cnn_layers(model, sample_image, device='cpu'):
    """可视化CNN各层"""
    visualizer = CNNLayerVisualizer(model)
    
    # 获取所有卷积层名称
    layer_names = visualizer.get_layer_names()
    print(f"可用层: {layer_names}")
    
    # 准备输入
    sample_image = sample_image.unsqueeze(0).to(device)
    
    # 可视化滤波器
    print("🔍 可视化卷积滤波器...")
    for layer_name in ['conv1', 'conv2', 'conv3']:
        if layer_name in layer_names:
            fig = visualizer.visualize_conv_filters(layer_name, max_filters=8)
            plt.savefig(f'filters_{layer_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 可视化特征图
    print("🔍 可视化特征图...")
    model.eval()
    with torch.no_grad():
        for layer_name in ['conv1', 'conv2', 'conv3']:
            if layer_name in layer_names:
                fig = visualizer.visualize_feature_maps(sample_image, layer_name, max_maps=8)
                plt.savefig(f'features_{layer_name}.png', dpi=300, bbox_inches='tight')
                plt.show()


def apply_grad_cam(model, sample_image, target_class=0, device='cpu'):
    """应用Grad-CAM"""
    # 找到最后一个卷积层
    last_conv_layer = None
    conv_layer_names = []
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
            conv_layer_names.append(name)
    
    if last_conv_layer is None:
        print("❌ 未找到卷积层")
        return
    
    print(f"📍 找到卷积层: {conv_layer_names}")
    print(f"📍 使用最后一层: {conv_layer_names[-1] if conv_layer_names else 'unknown'}")
    
    # 应用不同的CAM方法
    sample_image = sample_image.unsqueeze(0).to(device)
    
    # 比较不同的CAM方法
    fig = compare_cam_methods(model, last_conv_layer, sample_image, target_class=target_class)
    plt.savefig('cam_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数 - 在虚拟环境中运行"""
    print("🎯 CNN可视化教学项目 - PyTorch版本")
    print("=" * 50)
    
    # 检查虚拟环境
    import sys
    venv_path = sys.prefix
    if 'venv_linux' not in venv_path:
        print("⚠️  警告：当前不在venv_linux虚拟环境中")
        print("请运行：source venv_linux/bin/activate")
    else:
        print(f"✅ 虚拟环境已激活: {venv_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建数据集
    print("📊 创建数据集...")
    train_transform, test_transform = create_transforms()
    
    train_dataset = CatDogDataset('./datasets', train_transform, train=True, num_samples=200)
    test_dataset = CatDogDataset('./datasets', test_transform, train=False, num_samples=50)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"📊 训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    
    # 创建模型
    model = SimpleCNN(num_classes=2)
    print("🧠 模型结构:")
    print(model)
    
    # 训练模型
    print("🚀 开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, num_epochs=5, device=device
    )
    
    # 绘制训练历史
    print("📈 绘制训练历史...")
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 保存模型
    model_path = 'cat_dog_classifier_pytorch.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, model_path)
    print(f"💾 模型已保存: {model_path}")
    
    # 可视化示例
    print("🎨 开始可视化...")
    sample_image, sample_label = test_dataset[0]
    sample_image = sample_image.to(device)
    
    print(f"📄 示例图片标签: {'猫' if sample_label == 0 else '狗'}")
    
    # 可视化CNN层
    visualize_cnn_layers(model, sample_image, device=device)
    
    # 应用Grad-CAM
    apply_grad_cam(model, sample_image, target_class=sample_label, device=device)
    
    print("✅ 所有任务完成！")
    print("\n📋 输出文件:")
    print("- training_history_pytorch.png")
    print("- cat_dog_classifier_pytorch.pth")
    print("- filters_*.png")
    print("- features_*.png")
    print("- cam_comparison.png")


if __name__ == "__main__":
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch版本
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    
    main()