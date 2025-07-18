# CNN可视化教学项目 🎯

面向程序员的CNN神经网络可视化教学项目，支持PyTorch和PaddlePaddle双框架对比学习。

## 🚀 快速开始

### 1. 环境设置（必须步骤）

项目严格按照CLAUDE.md要求使用`venv_linux`虚拟环境：

```bash
# 克隆项目
git clone <repository-url>
cd cnn-visual-tutorial

# 一键设置虚拟环境
./scripts/setup_venv.sh

# 激活虚拟环境
source scripts/activate_venv.sh

# 验证虚拟环境
which python  # 应该显示项目路径下的venv_linux
```

### 2. 虚拟环境使用说明

**每次使用前必须激活虚拟环境：**
```bash
# 激活venv_linux虚拟环境
source venv_linux/bin/activate

# 或在任何目录下使用
source /path/to/cnn-visual-tutorial/scripts/activate_venv.sh
```

**虚拟环境中安装依赖：**
```bash
# 确保在虚拟环境中
source scripts/activate_venv.sh

# 安装/更新依赖
pip install -r requirements.txt

# 开发环境安装更多工具
pip install -r requirements-dev.txt
```

### 3. 运行示例

```bash
# 一键运行所有示例（自动激活虚拟环境）
./scripts/run_examples.sh

# 单独运行PyTorch示例
source scripts/activate_venv.sh
python examples/pytorch/cat_dog_classifier.py

# 单独运行PaddlePaddle示例
source scripts/activate_venv.sh
python examples/paddle/cat_dog_classifier.py

# 启动Jupyter Lab
./scripts/start_jupyter.sh
```

## 📁 项目结构

```
cnn-visual-tutorial/
├── venv_linux/                 # 虚拟环境目录（自动生成）
├── scripts/                    # 虚拟环境管理脚本
│   ├── setup_venv.sh          # 一键设置虚拟环境
│   ├── activate_venv.sh       # 激活虚拟环境
│   ├── run_examples.sh        # 运行示例
│   └── start_jupyter.sh       # 启动Jupyter
├── visualization/
│   ├── pytorch/               # PyTorch可视化工具
│   └── paddle/                # PaddlePaddle可视化工具
├── examples/
│   ├── pytorch/               # PyTorch示例
│   └── paddle/                # PaddlePaddle示例
├── notebooks/
│   └── codelab-01-visual-setup.ipynb
├── docs/markdown/
├── requirements.txt            # 生产环境依赖
├── requirements-dev.txt        # 开发环境依赖
└── .gitignore
```

## 🎯 核心功能

### 1. 完整可视化能力
- ✅ **卷积层滤波器可视化**：理解网络学习到的特征
- ✅ **特征图可视化**：逐层分析网络响应
- ✅ **类激活图（CAM）**：Grad-CAM、Grad-CAM++、Score-CAM
- ✅ **梯度可视化**：反向传播过程分析

### 2. 双框架支持
- **PyTorch 2.1+** 完整实现
- **PaddlePaddle 2.5+** 对应实现
- 代码结构一致，易于对比学习

### 3. 实战项目
- **猫狗分类**完整项目
- 包含训练、评估、可视化全流程
- 真实图片数据处理

## 📚 学习路径

### 新手入门
1. **环境准备**：按快速开始设置虚拟环境
2. **基础教程**：运行 `notebooks/codelab-01-visual-setup.ipynb`
3. **动手实践**：运行PyTorch和PaddlePaddle示例
4. **深入理解**：阅读 `docs/markdown/01-cnn-visualization.md`

### 进阶学习
1. **修改模型**：尝试不同CNN架构
2. **自定义数据**：替换为自己的图片数据集
3. **扩展可视化**：添加新的可视化方法
4. **性能优化**：使用GPU加速训练

## 🔧 虚拟环境管理

### 常用命令
```bash
# 激活虚拟环境
source venv_linux/bin/activate

# 检查是否在虚拟环境中
which python  # 应该显示项目路径

# 安装新包
pip install package_name

# 保存依赖
pip freeze > requirements.txt

# 退出虚拟环境
deactivate
```

### 开发环境设置
```bash
# 完整开发环境
source scripts/setup_venv.sh
pip install -r requirements-dev.txt

# 运行测试
source scripts/activate_venv.sh
pytest tests/
```

## 🐛 故障排除

### 虚拟环境问题
```bash
# 如果虚拟环境损坏，重新创建
rm -rf venv_linux
./scripts/setup_venv.sh

# 检查Python版本
python --version  # 应显示3.8+
```

### 依赖问题
```bash
# 更新pip
source scripts/activate_venv.sh
python -m pip install --upgrade pip

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

## 📖 文档导航

| 文档 | 内容 |
|------|------|
| `notebooks/codelab-01-visual-setup.ipynb` | 交互式入门教程 |
| `docs/markdown/01-cnn-visualization.md` | 完整可视化指南 |
| `examples/pytorch/cat_dog_classifier.py` | PyTorch完整示例 |
| `examples/paddle/cat_dog_classifier.py` | PaddlePaddle完整示例 |

## 🤝 贡献指南

1. **确保虚拟环境激活**：所有开发必须在`venv_linux`中进行
2. **代码规范**：使用`black`格式化，`flake8`检查
3. **测试**：运行`pytest tests/`确保通过
4. **文档**：更新相关markdown文档

## 📄 许可证

MIT License - 详见LICENSE文件

---

**⚠️ 重要提醒**：本项目严格按照CLAUDE.md要求，**必须使用`venv_linux`虚拟环境**。所有示例和脚本都设计为在虚拟环境中运行。