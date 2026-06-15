#!/bin/bash
# Batch 2: 四模型并行 (各占不同 GPU)
# 在 CatBoost 完成后运行
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "Batch 2: 四模型并行训练"
echo "  XGBoost: GPU 0-1"
echo "  MLP v1:  GPU 2"
echo "  MLP v2:  GPU 3"
echo "  TabM:    GPU 4-5"
echo "============================================"

# 并行启动四个子进程
CUDA_VISIBLE_DEVICES=0,1 python3 -u src/train_xgboost.py &
PID_XGB=$!

CUDA_VISIBLE_DEVICES=2   python3 -u -c "
from src.train_mlp import train
train(feature_set='full', save_prefix='mlp', resume=True, sample_rate=1)
" &
PID_MLP1=$!

CUDA_VISIBLE_DEVICES=3   python3 -u -c "
from src.data_utils import compute_tda_clusters
compute_tda_clusters()
from src.train_mlp import train
train(feature_set='tda', save_prefix='mlp', resume=True, sample_rate=1)
" &
PID_MLP2=$!

CUDA_VISIBLE_DEVICES=4,5 python3 -u -c "
from src.train_tabm import train
train()
" &
PID_TABM=$!

# 等待全部完成
echo "Waiting for all models..."
wait $PID_XGB  && echo "✅ XGBoost done"  || echo "❌ XGBoost failed"
wait $PID_MLP1 && echo "✅ MLP v1 done"   || echo "❌ MLP v1 failed"
wait $PID_MLP2 && echo "✅ MLP v2 done"   || echo "❌ MLP v2 failed"
wait $PID_TABM && echo "✅ TabM done"     || echo "❌ TabM failed"

echo "============================================"
echo "Batch 2 complete!"
echo "============================================"
