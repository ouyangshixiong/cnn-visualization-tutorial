"""
PaddlePaddle猫狗分类器 + 完整可视化
包含CNN模型、训练、评估和可视化
"""

import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
import paddle.vision.transforms as transforms
from paddle.io import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import cv2

# 导入可视化工具
import sys
sys.path.append('../..')
from visualization.paddle.layer_visualizer import CNNLayerVisualizer
from visualization.paddle.cam_visualizer import GradCAM, GradCAMPlusPlus, ScoreCAM, compare_cam_methods


class SimpleCNN(nn.Layer):
    """简单的CNN模型用于猫狗分类"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2D(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 假设输入224x224
        self.fc2 = nn.Linear(512, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一层
        x = self.pool(self.relu(self.conv1(x)))  # 32x112x112
        
        # 第二层
        x = self.pool(self.relu(self.conv2(x)))  # 64x56x56
        
        # 第三层
        x = self.pool(self.relu(self.conv3(x)))  # 128x28x28
        
        # 展平
        x = paddle.flatten(x, start_axis=1)
        
        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class CatDogDataset(Dataset):
    """猫狗数据集"""
    
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # 简化的文件列表 - 实际使用时需要完整的图片路径
        self.images = []
        self.labels = []
        
        # 这里使用随机生成的图片作为示例
        # 实际使用时应该加载真实的猫狗图片
        for i in range(100):
            if i % 2 == 0:
                self.images.append(f"cat_{i}.jpg")
                self.labels.append(0)  # 猫
            else:
                self.images.append(f"dog_{i}.jpg")
                self.labels.append(1)  # 狗
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 创建随机图片作为示例
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255.0
        label = self.labels[idx]
        
        return img, label


def create_transforms():
    """创建数据转换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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


def train_model(model, train_loader, val_loader, num_epochs=10):
    """训练模型"""
    device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_id, (images, labels) in enumerate(train_loader):
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            train_loss += loss.item()
            pred = nn.functional.softmax(outputs, axis=1)
            pred = paddle.argmax(pred, axis=1)
            train_correct += (pred == labels).numpy().sum()
            train_total += labels.shape[0]
        
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * train_correct / train_total)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with paddle.no_grad():
            for images, labels in val_loader:
                images = paddle.to_tensor(images)
                labels = paddle.to_tensor(labels)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                pred = nn.functional.softmax(outputs, axis=1)
                pred = paddle.argmax(pred, axis=1)
                val_correct += (pred == labels).numpy().sum()
                val_total += labels.shape[0]
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * val_correct / val_total)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_paddle.png')
    plt.show()


def visualize_cnn_layers(model, sample_image):
    """可视化CNN各层"""
    visualizer = CNNLayerVisualizer(model)
    
    # 获取所有卷积层名称
    layer_names = visualizer.get_layer_names()
    print(f"Available layers: {layer_names}")
    
    # 可视化第一个卷积层的滤波器
    if 'conv1' in layer_names:
        fig = visualizer.visualize_conv_filters('conv1')
        plt.savefig('conv1_filters_paddle.png')
        plt.show()
    
    # 可视化各层的特征图
    sample_image = paddle.to_tensor(sample_image)
    sample_image = sample_image.unsqueeze(0)
    
    with paddle.no_grad():
        for layer_name in layer_names[:3]:  # 只可视化前3层
            print(f"Visualizing feature maps for {layer_name}")
            fig = visualizer.visualize_feature_maps(sample_image, layer_name)
            plt.savefig(f'feature_maps_{layer_name}_paddle.png')
            plt.show()


def apply_grad_cam(model, sample_image, target_class=0):
    """应用Grad-CAM"""
    # 找到最后一个卷积层
    last_conv_layer = None
    for name, layer in model.named_sublayers():
        if isinstance(layer, nn.Conv2D):
            last_conv_layer = layer
    
    if last_conv_layer is None:
        print("No Conv2D layer found")
        return
    
    # 应用不同的CAM方法
    sample_image = paddle.to_tensor(sample_image)
    sample_image = sample_image.unsqueeze(0)
    
    # 比较不同的CAM方法
    fig = compare_cam_methods(model, last_conv_layer, sample_image)
    plt.savefig('cam_comparison_paddle.png')
    plt.show()


def main():
    """主函数"""
    # 设置随机种子
    paddle.seed(42)
    np.random.seed(42)
    
    # 创建数据转换
    train_transform, test_transform = create_transforms()
    
    # 创建数据集
    train_dataset = CatDogDataset('./datasets', train_transform, train=True)
    test_dataset = CatDogDataset('./datasets', test_transform, train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = SimpleCNN(num_classes=2)
    print(model)
    
    # 训练模型
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, num_epochs=5
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, 
                         train_accuracies, val_accuracies)
    
    # 保存模型
    paddle.save(model.state_dict(), 'cat_dog_classifier_paddle.pdparams')
    
    # 可视化示例
    print("Visualizing CNN layers...")
    sample_image, _ = test_dataset[0]
    visualize_cnn_layers(model, sample_image)
    
    # 应用Grad-CAM
    print("Applying Grad-CAM...")
    apply_grad_cam(model, sample_image)


if __name__ == "__main__":
    main()