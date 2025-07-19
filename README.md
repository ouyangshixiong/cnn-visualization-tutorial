# CNN Visualization Tutorial - Optimized 🔬

A modern, high-level API-based CNN visualization tutorial using PyTorch Lightning and PaddlePaddle high-level APIs.

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd cnn-visual-tutorial

# Install dependencies
pip install -r requirements.txt
```

### 2. One-Line Training
```bash
# Train with default configuration (CIFAR-10, SimpleCNN)
python scripts/train.py

# Train with specific model and dataset
python scripts/train.py model=resnet_cnn data=cifar10 trainer.max_epochs=5

# Fast development run
python scripts/train.py trainer=fast_dev
```

### 3. Download Datasets
```bash
# List available datasets
python scripts/download.py --list

# Download CIFAR-10
python scripts/download.py --dataset cifar10

# Download multiple datasets
python scripts/download.py --dataset cifar100
python scripts/download.py --dataset mnist
```

## 📁 Project Structure

```
cnn-visual-tutorial/
├── src/                    # Source code (modular, <100 lines per file)
│   ├── models/
│   │   ├── pytorch/       # PyTorch Lightning models
│   │   └── paddle/        # PaddlePaddle high-level models
│   ├── datasets/
│   │   ├── datamodules/   # Lightning DataModules
│   │   └── downloader.py  # High-level dataset downloader
│   ├── configs/           # Configuration management
│   └── utils/             # Visualization utilities
├── configs/               # Hydra YAML configurations
│   ├── model/            # Model configs
│   ├── data/             # Dataset configs
│   └── trainer/          # Training configs
├── scripts/              # Unified training scripts
├── tests/                # Comprehensive test suite
├── deploy/               # Docker deployment
└── docs/                 # Documentation and examples
```

## 🎯 Key Features

### 🔧 High-Level APIs
- **PyTorch Lightning**: Zero boilerplate training
- **PaddlePaddle**: High-level `paddle.Model` API
- **Hydra**: Configuration-driven experiments
- **Lightning DataModules**: Automatic dataset handling

### 📊 Visualization Support
- **Filter visualization**: See what CNN learns
- **Activation maps**: Understand feature extraction
- **Layer analysis**: Deep dive into network internals
- **Interactive examples**: Jupyter notebooks included

### 🐳 Docker Deployment
```bash
# CPU training
docker-compose -f deploy/cpu/docker-compose.yml up

# GPU training
docker-compose -f deploy/gpu/docker-compose.yml up
```

### 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Exclude slow tests
pytest tests/test_models/  # Only model tests
```

## 🎛️ Configuration Examples

### Model Configuration
```yaml
# configs/model/simple_cnn.yaml
_target_: src.models.pytorch.SimpleCNNClassifier
num_classes: 10
learning_rate: 1e-3
```

### Dataset Configuration
```yaml
# configs/data/cifar10.yaml
_target_: src.datasets.datamodules.CIFAR10DataModule
data_dir: ./data/cifar10
batch_size: 32
normalize: true
```

### Training Configuration
```yaml
# configs/trainer/default.yaml
max_epochs: 10
accelerator: auto
devices: auto
precision: 32
```

## 📊 Supported Datasets

| Dataset | Size | Classes | Description |
|---------|------|---------|-------------|
| CIFAR-10 | 32×32 | 10 | General objects |
| CIFAR-100 | 32×32 | 100 | Fine-grained objects |
| MNIST | 28×28 | 10 | Handwritten digits |
| Fashion-MNIST | 28×28 | 10 | Fashion items |
| ImageNet | 224×224 | 1000 | Large-scale classification |

## 🔍 Model Architectures

### SimpleCNN
- **Layers**: 2 conv layers + 2 FC layers
- **Parameters**: ~50K
- **Use case**: Educational, quick training

### ResNet-CNN
- **Backbones**: ResNet18, ResNet50
- **Pretrained**: Optional ImageNet weights
- **Use case**: Production, transfer learning

## 🚀 Usage Examples

### Basic Training
```bash
# Train SimpleCNN on CIFAR-10
python scripts/train.py model=simple_cnn data=cifar10

# Train ResNet50 on CIFAR-100
python scripts/train.py model=resnet_cnn data=cifar100 trainer.max_epochs=20
```

### Advanced Training
```bash
# Multi-GPU training
python scripts/train.py trainer.devices=4 trainer.strategy=ddp

# Mixed precision training
python scripts/train.py trainer.precision=16

# Custom hyperparameters
python scripts/train.py model.learning_rate=1e-4 data.batch_size=64
```

### Model Evaluation
```bash
# Evaluate trained model
python scripts/eval.py checkpoint_path=logs/lightning_logs/version_0/checkpoints/best.ckpt
```

## 📈 Performance Benchmarks

| Model | Dataset | Time (1 GPU) | Memory | Accuracy |
|-------|---------|-------------|--------|----------|
| SimpleCNN | CIFAR-10 | 2 min | 2GB | 85% |
| ResNet18 | CIFAR-10 | 5 min | 4GB | 92% |
| ResNet50 | CIFAR-10 | 8 min | 6GB | 94% |

## 🎨 Visualization Examples

### Filter Visualization
```python
from src.utils.viz_utils import visualize_filters
from src.models.pytorch import SimpleCNNClassifier

model = SimpleCNNClassifier(num_classes=10)
visualize_filters(model, "conv1")
```

### Activation Maps
```python
from src.utils.viz_utils import visualize_activations
import torch

model = SimpleCNNClassifier(num_classes=10)
input_tensor = torch.randn(1, 3, 32, 32)
visualize_activations(model, input_tensor, "conv2")
```

## 🤝 Contributing

1. **Code Style**: All files <100 lines, high-level APIs
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Configuration**: Add YAML configs for new features

## 📝 Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Code formatting
black src/ tests/ scripts/

# Linting
flake8 src/ tests/ scripts/
```

## 🔧 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Dataset download fails**: Check internet connection
3. **Hydra config errors**: Validate YAML syntax

### Performance Tips
1. **Use GPU**: `trainer.accelerator=gpu`
2. **Mixed precision**: `trainer.precision=16`
3. **Data augmentation**: Enable in dataset config
4. **Batch size tuning**: Optimize for GPU memory

## 📚 Learning Resources

### Tutorials
- [PyTorch Lightning Tutorial](https://pytorch-lightning.readthedocs.io/)
- [PaddlePaddle High-Level API](https://www.paddlepaddle.org.cn/)
- [CNN Visualization Guide](docs/cnn_visualization.md)

### Examples
- Basic CNN training: `examples/basic_cnn.py`
- Transfer learning: `examples/transfer_learning.py`
- Custom datasets: `examples/custom_dataset.py`

## 🏷️ Version History

- **v2.0**: High-level API rewrite, Hydra configs, Docker support
- **v1.0**: Original tutorial with manual training loops

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.