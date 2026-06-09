# Jane Street — GPU 环境一键安装 (Windows PowerShell)
# ======================================================
# 自动检测 CUDA 版本，安装对应 PyTorch + 依赖
# 适用: Windows 10/11, NVIDIA GPU (RTX 3090 等), CUDA 11.x 或 12.x

$ErrorActionPreference = "Stop"
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Jane Street GPU 环境安装 (Windows)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ============================================================
# 1. 检测 CUDA 版本
# ============================================================
Write-Host "`n[1/4] 检测 CUDA 版本..." -ForegroundColor Yellow

$cudaVersion = $null
try {
    $nvidiaOutput = & nvidia-smi 2>$null | Out-String
    Write-Host $nvidiaOutput
    if ($nvidiaOutput -match 'CUDA Version:\s*(\d+\.\d+)') {
        $cudaVersion = $matches[1]
        Write-Host "`n  nvidia-smi 报告 CUDA: $cudaVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "  nvidia-smi 不可用" -ForegroundColor Red
}

if (-not $cudaVersion) {
    # 尝试 nvcc
    try {
        $nvccOutput = & nvcc --version 2>$null | Out-String
        if ($nvccOutput -match 'release\s+(\d+\.\d+)') {
            $cudaVersion = $matches[1]
            Write-Host "  nvcc 报告 CUDA: $cudaVersion" -ForegroundColor Green
        }
    } catch {
        Write-Host "  无法检测 CUDA 版本，假设 CUDA 12.x" -ForegroundColor Yellow
        $cudaVersion = "12.0"
    }
}

# ============================================================
# 2. 创建虚拟环境
# ============================================================
Write-Host "`n[2/4] 创建虚拟环境..." -ForegroundColor Yellow

# 查找 python 命令 (Windows 上可能是 python 或 python3)
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "  错误: 未找到 Python！请先安装 Python 3.10+" -ForegroundColor Red
    Write-Host "  下载: https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  ⚠ 安装时务必勾选 'Add Python to PATH'" -ForegroundColor Red
    exit 1
}
Write-Host "  使用: $pythonCmd" -ForegroundColor Green

if (-not (Test-Path "venv")) {
    & $pythonCmd -m venv venv
    Write-Host "  虚拟环境已创建: venv\" -ForegroundColor Green
} else {
    Write-Host "  虚拟环境已存在，跳过" -ForegroundColor Green
}

# 激活虚拟环境
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "  激活虚拟环境..." -ForegroundColor Green
    . $activateScript
} else {
    Write-Host "  错误: 找不到 $activateScript" -ForegroundColor Red
    exit 1
}

# 升级 pip
Write-Host "  升级 pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# ============================================================
# 3. 安装 PyTorch
# ============================================================
Write-Host "`n[3/4] 安装 PyTorch..." -ForegroundColor Yellow

$cudaMajor = [int]($cudaVersion -split '\.')[0]

if ($cudaMajor -ge 12) {
    Write-Host "  检测到 CUDA 12.x → 安装 PyTorch (cu124)" -ForegroundColor Green
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} elseif ($cudaMajor -ge 11) {
    Write-Host "  检测到 CUDA 11.x → 安装 PyTorch (cu118)" -ForegroundColor Green
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "  CUDA < 11，安装 CPU 版 PyTorch" -ForegroundColor Yellow
    python -m pip install torch torchvision torchaudio
}

# ============================================================
# 4. 安装其他依赖
# ============================================================
Write-Host "`n[4/4] 安装其他依赖..." -ForegroundColor Yellow
python -m pip install -r requirements_gpu.txt

# ============================================================
# 验证
# ============================================================
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "  安装完成！验证:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

python -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}, VRAM {p.total_memory/1024**3:.1f} GB')
print()

import catboost
print(f'CatBoost: {catboost.__version__}')

import xgboost as xgb
print(f'XGBoost: {xgb.__version__}')

import polars as pl
print(f'Polars: {pl.__version__}')

print()
print('所有包安装成功！')
"@

Write-Host "`n接下来运行预检查:" -ForegroundColor Cyan
Write-Host "  python check_env.py" -ForegroundColor White
Write-Host "`n一切正常后，开始训练:" -ForegroundColor Cyan
Write-Host "  python train_all.py" -ForegroundColor White
Write-Host "`n(训练日志自动保存到 models/training_report.json)" -ForegroundColor Gray
