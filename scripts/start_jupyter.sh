#\!/bin/bash
# 启动Jupyter Lab（自动激活虚拟环境）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 激活虚拟环境
source "$PROJECT_ROOT/venv_linux/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "📚 启动Jupyter Lab..."
echo "=" 30
echo "✅ 虚拟环境: $VIRTUAL_ENV"
echo "📍 Python: $(which python)"

# 检查Jupyter是否安装
if \! command -v jupyter-lab >/dev/null 2>&1; then
    echo "📦 安装Jupyter Lab..."
    pip install jupyterlab
fi

# 启动Jupyter Lab
cd "$PROJECT_ROOT"
jupyter lab notebooks/ --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF < /dev/null