# Docker 部署配置

## GPU 部署（CUDA 12.6 + Ubuntu 24.04）

### 系统要求
- NVIDIA GPU驱动版本：>= 535.x
- Docker版本：>= 19.03
- NVIDIA Docker运行时：>= 2.0

### 快速开始
```bash
# 构建GPU镜像
docker-compose -f deploy/gpu/docker-compose.yml build

# 启动GPU容器
docker-compose -f deploy/gpu/docker-compose.yml up

# 使用GPU训练
docker-compose -f deploy/gpu/docker-compose.yml exec cnn-gpu python scripts/train.py model=resnet18 data=cifar10 trainer.max_epochs=10
```

### CUDA环境信息
- **CUDA版本**: 12.6
- **Ubuntu版本**: 24.04 LTS
- **PyTorch版本**: 2.6.0
- **Python版本**: 3.10

### 多GPU训练
```bash
# 自动检测所有可用GPU
docker-compose -f deploy/gpu/docker-compose.yml exec cnn-gpu python scripts/train.py trainer.devices=auto trainer.strategy=ddp

# 指定GPU数量
docker-compose -f deploy/gpu/docker-compose.yml exec cnn-gpu python scripts/train.py trainer.devices=4 trainer.strategy=ddp
```

### GPU监控
```bash
# 查看GPU使用情况
docker-compose -f deploy/gpu/docker-compose.yml exec cnn-gpu nvidia-smi

# 实时监控
docker-compose -f deploy/gpu/docker-compose.yml exec cnn-gpu watch -n 1 nvidia-smi
```

## CPU部署

### 快速开始
```bash
# 构建CPU镜像
docker-compose -f deploy/cpu/docker-compose.yml build

# 启动CPU容器
docker-compose -f deploy/cpu/docker-compose.yml up
```

### 故障排除

#### GPU驱动问题
如果容器无法识别GPU，请检查：
1. NVIDIA驱动是否安装正确：`nvidia-smi`
2. Docker NVIDIA运行时是否安装：`docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`
3. 重启Docker服务：`sudo systemctl restart docker`

#### 权限问题
如果遇到权限错误：
```bash
# 将用户加入docker组
sudo usermod -aG docker $USER
# 重新登录或重启系统
```

#### 存储问题
数据将保存在以下目录：
- `./data/` - 数据集缓存
- `./logs/` - 训练日志和检查点