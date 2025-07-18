#\!/bin/bash
# 运行示例脚本（自动激活虚拟环境）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 激活虚拟环境
source "$PROJECT_ROOT/venv_linux/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "🎯 运行CNN可视化教学项目示例..."
echo "=" 50

# 检查虚拟环境
echo "✅ 虚拟环境: $VIRTUAL_ENV"
echo "📍 Python: $(which python)"

# 创建数据集目录
mkdir -p "$PROJECT_ROOT/datasets"

# 运行PyTorch示例
echo ""
echo "🚀 运行PyTorch示例..."
cd "$PROJECT_ROOT"
python examples/pytorch/cat_dog_classifier.py

# 运行PaddlePaddle示例
echo ""
echo "🚀 运行PaddlePaddle示例..."
python examples/paddle/cat_dog_classifier.py

echo ""
echo "✅ 所有示例运行完成！"
echo "📁 查看生成的可视化文件:"
ls -la *.png 2>/dev/null || echo "无可视化文件生成"
EOF < /dev/null