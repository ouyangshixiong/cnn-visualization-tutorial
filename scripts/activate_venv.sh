#!/bin/bash
# 激活虚拟环境脚本
# 使用方法：source scripts/activate_venv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查虚拟环境是否存在
if [ ! -d "$PROJECT_ROOT/venv_linux" ]; then
    echo "❌ 虚拟环境不存在，请先运行: ./scripts/setup_venv.sh"
    return 1
fi

# 激活虚拟环境
source "$PROJECT_ROOT/venv_linux/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PROJECT_ROOT="$PROJECT_ROOT"

# 检查是否成功激活
if [[ "$VIRTUAL_ENV" == *"venv_linux"* ]]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
    echo "📍 Python路径: $(which python)"
    echo "📍 Python版本: $(python --version)"
    echo "📍 项目根目录: $PROJECT_ROOT"
else
    echo "❌ 虚拟环境激活失败"
    return 1
fi