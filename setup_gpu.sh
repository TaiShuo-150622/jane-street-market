#!/bin/bash
# Jane Street — GPU 环境一键安装
# ===============================
# 自动检测 CUDA 版本，安装对应 PyTorch + 依赖
# 适用于: 6 × 2080 Ti, CUDA 11.8 或 12.4

set -e

echo "================================================"
echo "  Jane Street GPU 环境安装"
echo "================================================"

# 1. 检测 CUDA 版本
echo ""
echo "[1/4] 检测 CUDA 版本..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "unknown")
    echo ""
    echo "  nvidia-smi 报告 CUDA: $CUDA_VERSION"
else
    echo "  ⚠ nvidia-smi 不可用，尝试 nvcc..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || echo "unknown")
        echo "  nvcc 报告 CUDA: $CUDA_VERSION"
    else
        echo "  无法检测 CUDA 版本，假设 CUDA 12.x"
        CUDA_VERSION="12.0"
    fi
fi

# 2. 创建虚拟环境
echo ""
echo "[2/4] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  虚拟环境已创建: venv/"
else
    echo "  虚拟环境已存在，跳过"
fi

source venv/bin/activate
pip install --upgrade pip

# 3. 安装 PyTorch
echo ""
echo "[3/4] 安装 PyTorch..."

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    # CUDA 12.x → cu124 (PyTorch 2.5+)
    echo "  检测到 CUDA 12.x → 安装 PyTorch (cu124)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [ "$CUDA_MAJOR" -ge 11 ]; then
    # CUDA 11.x → cu118
    echo "  检测到 CUDA 11.x → 安装 PyTorch (cu118)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  ⚠ CUDA 版本 < 11，安装 CPU 版 PyTorch"
    pip install torch torchvision torchaudio
fi

# 4. 安装其他依赖
echo ""
echo "[4/4] 安装其他依赖..."
pip install -r requirements_gpu.txt

# 验证
echo ""
echo "================================================"
echo "  安装完成！验证:"
echo "================================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}, {p.total_mem/1024**3:.1f} GB')
print()

import catboost
print(f'CatBoost: {catboost.__version__}')

import xgboost as xgb
print(f'XGBoost: {xgb.__version__}')

import polars as pl
print(f'Polars: {pl.__version__}')

print()
print('所有包安装成功！')
"

echo ""
echo "接下来运行预检查:"
echo "  python check_env.py"
echo ""
echo "一切正常后，开始训练:"
echo "  nohup python -u train_all.py > train_\$(date +%Y%m%d_%H%M).log 2>&1 &"
