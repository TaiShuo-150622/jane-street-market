# GPU 机器部署指南

## 环境: 6 × 2080 Ti, CUDA 11.8 或 12.4

---

## 方式 A：GitHub 全流程（推荐）

### 你在本地准备

```bash
# 1. 初始化 git + 推送代码
cd /Users/taishuo/codebuddy/Jane_Street_Market
git init
git add -A
git commit -m "GPU training: CatBoost + MLP + XGBoost + ensemble"
git remote add origin git@github.com:你的用户名/jane-street.git
git push -u origin main

# 2. 分包数据（本地运行一次）
bash split_data_for_release.sh
# 生成 data_chunks/ 目录，含 ~6 个 2GB 文件 + checksums.md5

# 3. 在 GitHub 上创建 Release
#    https://github.com/你的用户名/jane-street/releases/new
#    Tag: v1.0
#    把 data_chunks/ 下所有文件拖进去上传
```

### 对方在 GPU 机器上操作

```bash
# 1. Clone 代码
git clone https://github.com/你的用户名/jane-street.git
cd jane-street

# 2. 下载数据
bash download_data.sh https://github.com/你的用户名/jane-street/releases/tag/v1.0

# 3. 安装环境
bash setup_gpu.sh
source venv/bin/activate

# 4. 预检查
python check_env.py

# 5. 开跑
tmux new -s jane
python train_all.py
```

---

## 方式 B：rsync 直传（有 SSH 时最快）

```bash
# 本地：代码秒传
rsync -avz --exclude='data/processed' --exclude='models' \
    /Users/taishuo/codebuddy/Jane_Street_Market/ \
    user@gpu:/path/to/Jane_Street_Market/

# 本地：数据断点续传
rsync -avP /Users/taishuo/codebuddy/Jane_Street_Market/data/processed/train_processed.parquet \
    user@gpu:/path/to/Jane_Street_Market/data/processed/
```

---

## 方式 C：网盘

代码打包 zip → 网盘分享。数据 11GB 分包（`split -b 2G` 切 6 块）→ 网盘分享。

---

## 恢复 / 重跑

```bash
# 中断后重启（自动跳过已完成的模型，MLP 从 checkpoint 续训）
python train_all.py

# 完全重来
rm -rf models/*.pth models/*.cbm models/*.json models/mlp_cache/
python train_all.py
```

---

## 常见问题

### Q: setup_gpu.sh 报错 "CUDA not detected"
A: 手动运行 `nvidia-smi` 确认驱动正常，然后按 CUDA 版本手动装 PyTorch：
```bash
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.x
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_gpu.txt
```

### Q: 训练中 OOM
A: 编辑 `train_all.py`，提高 `TREE_SAMPLE_RATE=8`（减少树模型数据量）。

### Q: CatBoost 报 "GPU training is not supported"
A: `pip uninstall catboost; pip install catboost`（确保是 GPU 版）

### Q: 下载的 parquet 校验失败
A: 重新下载损坏的分块文件（checksums.md5 会告诉你是哪个）
