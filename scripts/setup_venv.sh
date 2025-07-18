#!/bin/bash

# CNN可视化教学项目 - 虚拟环境设置脚本
# 按照CLAUDE.md要求使用venv_linux虚拟环境

echo "🚀 开始设置CNN可视化教学项目虚拟环境..."

# 创建虚拟环境目录
if [ ! -d "venv_linux" ]; then
    echo "📁 创建虚拟环境目录..."
    python3 -m venv venv_linux
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv_linux/bin/activate

# 升级pip
echo "⬆️  升级pip..."
python -m pip install --upgrade pip

# 安装依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 安装开发依赖（如果存在）
if [ -f "requirements-dev.txt" ]; then
    echo "📦 安装开发依赖..."
    pip install -r requirements-dev.txt
fi

# 创建激活脚本
cat > scripts/activate_venv.sh << 'EOF'
#!/bin/bash
# 激活虚拟环境
source venv_linux/bin/activate
echo "✅ 虚拟环境已激活"
echo "当前Python路径: $(which python)"
EOF

chmod +x scripts/activate_venv.sh

# 创建运行示例脚本
cat > scripts/run_examples.sh << 'EOF'
#!/bin/bash
# 运行示例脚本（自动激活虚拟环境）

source venv_linux/bin/activate

echo "🎯 运行PyTorch示例..."
python examples/pytorch/cat_dog_classifier.py

echo "🎯 运行PaddlePaddle示例..."
python examples/paddle/cat_dog_classifier.py

echo "✅ 示例运行完成"
EOF

chmod +x scripts/run_examples.sh

# 创建启动Jupyter脚本
cat > scripts/start_jupyter.sh << 'EOF'
#!/bin/bash
# 启动Jupyter Lab（自动激活虚拟环境）

source venv_linux/bin/activate

echo "📚 启动Jupyter Lab..."
jupyter lab notebooks/
EOF

chmod +x scripts/start_jupyter.sh

echo "✅ 虚拟环境设置完成！"
echo ""
echo "📋 使用方法:"
echo "1. 激活虚拟环境: source scripts/activate_venv.sh"
echo "2. 运行示例: ./scripts/run_examples.sh"
echo "3. 启动Jupyter: ./scripts/start_jupyter.sh"
echo ""
echo "🔍 验证虚拟环境:"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"