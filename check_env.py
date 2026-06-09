#!/usr/bin/env python3
"""
环境检查脚本
============
在正式训练前运行，逐一验证:
1. Python 版本
2. CUDA / GPU 可用性
3. 关键包版本
4. 数据文件存在且可读
5. 磁盘空间
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)  # 确保工作目录正确

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

issues = []

print("=" * 60)
print("  Jane Street — 环境检查")
print("=" * 60)

# 1. Python 版本
print(f"\n[1] Python: {sys.version}")
if sys.version_info < (3, 10):
    issues.append("Python < 3.10")
    print(f"  {FAIL} 需要 Python 3.10+")

# 2. CUDA / GPU
print(f"\n[2] CUDA / GPU:")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  PyTorch CUDA: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available} {PASS if cuda_available else FAIL}")
    if cuda_available:
        n_gpus = torch.cuda.device_count()
        total_vram = 0
        print(f"  GPU 数量: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024**3
            total_vram += vram_gb
            print(f"  GPU {i}: {props.name}, {vram_gb:.1f} GB VRAM, CC={props.major}.{props.minor}")
            if vram_gb < 8:
                issues.append(f"GPU {i} VRAM < 8GB ({vram_gb:.1f} GB)")
        print(f"  总 VRAM: {total_vram:.1f} GB  {PASS if total_vram > 16 else WARN}")
    else:
        issues.append("CUDA 不可用")
except ImportError:
    print(f"  {FAIL} PyTorch 未安装")
    issues.append("PyTorch 未安装")

# 3. 关键包
print(f"\n[3] 关键包:")
for pkg, min_ver in [
    ("polars", "1.0.0"),
    ("numpy", "1.24.0"),
    ("catboost", "1.2.0"),
    ("xgboost", "2.0.0"),
    ("sklearn", "1.3.0"),
    ("scipy", "1.10.0"),
]:
    try:
        mod = __import__(pkg)
        ver = mod.__version__
        status = PASS
        print(f"  {pkg}: {ver} {status}")
    except ImportError:
        print(f"  {pkg}: 未安装 {FAIL}")
        issues.append(f"{pkg} 未安装")

# 4. 数据文件
print(f"\n[4] 数据文件:")
data_path = Path("data/processed/train_processed.parquet")
if data_path.exists():
    size_gb = data_path.stat().st_size / 1e9
    print(f"  {data_path}: {size_gb:.2f} GB {PASS}")

    # 测试读取
    import polars as pl
    try:
        lazy = pl.scan_parquet(data_path)
        n = lazy.select(pl.len()).collect().item()
        n_cols = len(lazy.collect_schema())
        print(f"  行数: {n:,}, 列数: {n_cols} {PASS}")
    except Exception as e:
        print(f"  读取失败: {e} {FAIL}")
        issues.append("数据文件读取失败")
else:
    print(f"  {data_path}: 不存在 {FAIL}")
    issues.append("数据文件不存在")

# 5. 磁盘空间
print(f"\n[5] 磁盘空间:")
import shutil
disk = shutil.disk_usage(Path.cwd())
free_gb = disk.free / 1e9
print(f"  可用: {free_gb:.1f} GB")
if free_gb < 20:
    print(f"  {WARN} 空间不足 20GB，模型文件可能存不下")
    issues.append("磁盘空间 < 20GB")

# 6. 系统 RAM
print(f"\n[6] 系统 RAM:")
try:
    import psutil
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    avail_gb = mem.available / 1e9
    print(f"  总量: {total_gb:.1f} GB, 可用: {avail_gb:.1f} GB")
    if total_gb < 16:
        print(f"  {FAIL} RAM < 16GB，MLP 训练可能 OOM")
        issues.append("系统 RAM < 16GB")
    elif total_gb < 32:
        print(f"  {WARN} RAM < 32GB，MLP 训练可能用 memmap 较慢")
except ImportError:
    print(f"  (psutil 未安装)")

# ---- 结论 ----
print(f"\n{'=' * 60}")
if issues:
    print(f"  发现问题 ({len(issues)}):")
    for i in issues:
        print(f"    {FAIL} {i}")
    print(f"\n  请先解决以上问题再运行 train_all.py")
else:
    print(f"  一切就绪！运行:")
    # 自动检测平台，给对应的命令提示
    import platform
    if platform.system() == "Windows":
        print(f"    python train_all.py")
        print(f"    (训练日志自动保存到 models/training_report.json)")
    else:
        print(f"    nohup python -u train_all.py > train_$(date +%Y%m%d_%H%M).log 2>&1 &")
print("=" * 60)
