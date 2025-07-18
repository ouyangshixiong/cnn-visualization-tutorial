# CNN神经网络可视化完全指南

## 概述

本指南将教你如何通过可视化技术深入理解CNN神经网络的工作原理。我们将重点介绍卷积层、特征图、类激活图等关键概念的可视化方法。

## 学习目标

- [ ] 理解CNN各层的作用和可视化原理
- [ ] 掌握卷积层滤波器和特征图的可视化
- [ ] 学会使用Grad-CAM等类激活图技术
- [ ] 比较PyTorch和PaddlePaddle的可视化差异

## 1. CNN基础可视化

### 1.1 卷积层滤波器可视化

卷积层滤波器（也称卷积核）是CNN学习到的特征检测器。通过可视化这些滤波器，我们可以理解网络在寻找什么样的特征。

#### 可视化方法

**PyTorch实现:**
```python
from visualization.pytorch.layer_visualizer import CNNLayerVisualizer

# 创建可视化器
visualizer = CNNLayerVisualizer(model)

# 可视化第一个卷积层的滤波器
fig = visualizer.visualize_conv_filters('conv1', max_filters=16)
plt.show()
```

**PaddlePaddle实现:**
```python
from visualization.paddle.layer_visualizer import CNNLayerVisualizer

# 创建可视化器
visualizer = CNNLayerVisualizer(model)

# 可视化第一个卷积层的滤波器
fig = visualizer.visualize_conv_filters('conv1', max_filters=16)
plt.show()
```

#### 结果解读

- **低层滤波器**：通常检测边缘、颜色、纹理等基础特征
- **中层滤波器**：检测更复杂的图案和形状
- **高层滤波器**：检测物体部件或整体结构

### 1.2 特征图可视化

特征图显示了卷积层对输入图片的响应，帮助我们理解网络如何逐层提取特征。

#### 单层特征图可视化

**PyTorch:**
```python
# 可视化特定层的特征图
fig = visualizer.visualize_feature_maps(input_tensor, 'conv2', max_maps=16)
```

**PaddlePaddle:**
```python
# 可视化特定层的特征图
fig = visualizer.visualize_feature_maps(input_tensor, 'conv2', max_maps=16)
```

#### 交互式特征图可视化

使用Plotly创建交互式可视化：

```python
# 交互式可视化
fig = visualizer.visualize_feature_maps_interactive(input_tensor, 'conv2')
fig.show()
```

### 1.3 梯度可视化

梯度可视化帮助我们理解反向传播过程中梯度的流动情况。

**使用方法:**
```python
# 可视化梯度
fig = visualizer.visualize_gradients(input_tensor, target_class, 'conv2')
```

## 2. 类激活图（CAM）技术

### 2.1 Grad-CAM

Grad-CAM通过计算目标类别对卷积层特征图的梯度权重，生成类激活图。

**PyTorch实现:**
```python
from visualization.pytorch.cam_visualizer import GradCAM

# 创建Grad-CAM
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate_cam(input_tensor, target_class=0)
```

**PaddlePaddle实现:**
```python
from visualization.paddle.cam_visualizer import GradCAM

# 创建Grad-CAM
gradcam = GradCAM(model, target_layer)
cam = gradcam.generate_cam(input_tensor, target_class=0)
```

### 2.2 Grad-CAM++

Grad-CAM++改进了Grad-CAM在多个目标实例情况下的表现。

**使用方法:**
```python
from visualization.pytorch.cam_visualizer import GradCAMPlusPlus

gradcam_pp = GradCAMPlusPlus(model, target_layer)
cam = gradcam_pp.generate_cam(input_tensor, target_class=0)
```

### 2.3 Score-CAM

Score-CAM通过扰动输入来评估每个特征图的重要性。

**使用方法:**
```python
from visualization.pytorch.cam_visualizer import ScoreCAM

scorecam = ScoreCAM(model, target_layer)
cam = scorecam.generate_cam(input_tensor, target_class=0)
```

### 2.4 CAM方法对比

可以使用内置函数比较不同CAM方法的效果：

```python
from visualization.pytorch.cam_visualizer import compare_cam_methods

# 比较三种CAM方法
fig = compare_cam_methods(model, target_layer, input_tensor, original_image)
plt.show()
```

## 3. 实际应用案例：猫狗分类

### 3.1 数据集准备

我们使用猫狗分类作为示例任务，展示完整的可视化流程。

**数据加载：**
```python
# 示例数据加载
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3.2 模型定义

**PyTorch模型:**
```python
class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**PaddlePaddle模型:**
```python
class CatDogCNN(nn.Layer):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2D(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2D(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = paddle.flatten(x, start_axis=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 3.3 完整可视化流程

**1. 训练模型**
```python
# 训练代码详见 examples/pytorch/cat_dog_classifier.py
# 或 examples/paddle/cat_dog_classifier.py
```

**2. 可视化各层**
```python
# 1. 卷积层滤波器可视化
fig1 = visualizer.visualize_conv_filters('conv1')
fig2 = visualizer.visualize_conv_filters('conv2')
fig3 = visualizer.visualize_conv_filters('conv3')
```

**3. 特征图可视化**
```python
# 2. 各层特征图对比
fig1 = visualizer.visualize_feature_maps(cat_tensor, 'conv1')
fig2 = visualizer.visualize_feature_maps(cat_tensor, 'conv2')
fig3 = visualizer.visualize_feature_maps(cat_tensor, 'conv3')
```

**4. 类激活图**
```python
# 3. 使用Grad-CAM分析网络关注区域
gradcam = GradCAM(model, model.conv3)
cam = gradcam.visualize_cam(cat_tensor, target_class=0, original_image=cat_img)
```

## 4. PyTorch vs PaddlePaddle对比

### 4.1 API差异

| 特性 | PyTorch | PaddlePaddle |
|------|---------|--------------|
| 动态图 | ✅ 原生支持 | ✅ 原生支持 |
| Hook注册 | `register_forward_hook` | `register_forward_post_hook` |
| 梯度计算 | `.backward()` | `.backward()` |
| 张量操作 | `.detach()` | `.detach()` |
| 模型保存 | `torch.save()` | `paddle.save()` |

### 4.2 性能对比

在相同的硬件条件下，两个框架的可视化性能表现相近。主要差异在于：

- **内存使用**：PaddlePaddle在某些场景下内存使用更优化
- **GPU支持**：两者都支持GPU加速
- **API设计**：PaddlePaddle的API更接近PyTorch，易于迁移

## 5. 高级可视化技巧

### 5.1 特征图动画

创建训练过程中特征图变化的动画：

```python
import matplotlib.animation as animation

fig, ax = plt.subplots()
im = ax.imshow(initial_feature_map, animated=True)

def animate(i):
    # 更新特征图
    im.set_array(feature_maps[i])
    return im,

ani = animation.FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()
```

### 5.2 3D特征可视化

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(w), range(h))
ax.plot_surface(X, Y, feature_map, cmap='viridis')
plt.show()
```

## 6. 常见问题解答

### Q1: 可视化结果不显示怎么办？
A: 检查是否正确调用了`plt.show()`，或在Jupyter环境中使用`%matplotlib inline`

### Q2: 内存不足怎么办？
A: 减少`max_filters`或`max_maps`参数，或使用更小的输入图片

### Q3: 如何选择目标层？
A: 通常选择最后一个卷积层用于Grad-CAM，所有卷积层都可以用于特征图可视化

## 7. 下一步学习

完成本指南后，建议继续学习：

1. [02-convolution-layers.md](02-convolution-layers.md) - 深入理解卷积操作
2. [03-feature-map-analysis.md](03-feature-map-analysis.md) - 高级特征图分析
3. [04-cam-techniques.md](04-cam-techniques.md) - 更多CAM技术
4. [05-comparison-guide.md](05-comparison-guide.md) - 框架对比分析

## 8. 代码示例运行

```bash
# 运行PyTorch示例
python examples/pytorch/cat_dog_classifier.py

# 运行PaddlePaddle示例
python examples/paddle/cat_dog_classifier.py

# 启动Jupyter教程
jupyter lab notebooks/
```

## 9. 资源链接

- [PyTorch官方文档](https://pytorch.org/docs/)
- [PaddlePaddle官方文档](https://www.paddlepaddle.org.cn/documentation)
- [Grad-CAM论文](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++论文](https://arxiv.org/abs/1710.11063)
- [Score-CAM论文](https://arxiv.org/abs/1910.01279)