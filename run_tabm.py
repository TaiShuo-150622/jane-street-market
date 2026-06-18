#!/usr/bin/env python3
"""
TabM 全卡训练 — 6×2080 Ti 全部用于 TabM
==========================================
pip install tabm polars numpy
python run_tabm.py
"""
import os, sys, time, json, gc
from pathlib import Path
import numpy as np
import polars as pl
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from src.data_utils import MODELS_DIR, TRAIN_PATH, TARGET_COL, WEIGHT_COL, TRAIN_END_DATE, ensure_models_dir
from src.metrics import weighted_r2

ensure_models_dir()

# ============================================================
# GPU 检测
# ============================================================
n_gpus = torch.cuda.device_count()
print(f"CUDA GPUs: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB, {p.multi_processor_count} SMs)")

if n_gpus < 1:
    print("ERROR: No GPU!")
    sys.exit(1)

# ============================================================
# 数据加载
# ============================================================
FEATURE_COLS = [f"feature_{i:02d}" for i in range(79) if i not in (9, 10, 11)]
CAT_FEATURE_COLS = ["feature_09", "feature_10", "feature_11"]
RESPONDER_COLS = [f"responder_{i}" for i in range(9) if i != 6]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]

print(f"\n加载数据: {TRAIN_PATH}")
# lazy 加载避免 OOM: 先过滤 train/val 再分别 collect
train_df = pl.scan_parquet(TRAIN_PATH).filter(pl.col("date_id") <= TRAIN_END_DATE).collect()
print(f"  Train 加载完成: {train_df.height:,} 行")

avail = [c for c in FEATURE_COLS + CAT_FEATURE_COLS + RESPONDER_COLS + LAG_COLS
         if c in train_df.columns]

X_train = train_df.select(avail).to_numpy().astype(np.float32)
y_train = train_df[TARGET_COL].to_numpy().astype(np.float32).ravel()
w_train = train_df[WEIGHT_COL].to_numpy().astype(np.float32).ravel()
del train_df  # 释放内存

val_df   = pl.scan_parquet(TRAIN_PATH).filter(pl.col("date_id") > TRAIN_END_DATE).collect()
print(f"  Val   加载完成: {val_df.height:,} 行")

X_val = val_df.select(avail).to_numpy().astype(np.float32)
y_val = val_df[TARGET_COL].to_numpy().astype(np.float32).ravel()
w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float32).ravel()
del val_df  # 释放内存

X_train = np.nan_to_num(X_train, nan=0.0)
X_val   = np.nan_to_num(X_val, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
y_val   = np.nan_to_num(y_val, nan=0.0)

print(f"Train: {X_train.shape[0]:,} × {X_train.shape[1]}  ({X_train.nbytes/1e9:.1f} GB)")
print(f"Val:   {X_val.shape[0]:,} × {X_val.shape[1]}")

# ============================================================
# TabM 训练 (全部 6 卡)
# ============================================================
from tabm import TabMRegressor

print(f"\nTabM 参数:")
print(f"  device: cuda (全部 {n_gpus} 卡)")
print(f"  训练行数: {len(X_train):,}")

t0 = time.time()

model = TabMRegressor(
    device='cuda',
    n_estimators=4,            # 单卡 24GB 保守 K=4
    batch_size=256,            # 控制内存
    random_state=42,
    verbose=True,
)

# 心跳线程：每 30 秒打印一次
import threading
running = True
def heartbeat():
    start = time.time()
    while running:
        time.sleep(30)
        if running:
            elapsed = time.time() - start
            mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"  [心跳] 训练中... {elapsed/60:.0f}min  GPU内存: {mem:.1f}GB", flush=True)

hb = threading.Thread(target=heartbeat, daemon=True)
hb.start()

print(f"  GPU 内存 (训前): {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
print(f"  训练中... (47M行 × {X_train.shape[1]}特征, 预计 30-90 min)")
model.fit(X_train, y_train)

running = False
train_time = time.time() - t0
print(f"  GPU 内存 (训后): {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
print(f"  训练完成: {train_time:.0f}s ({train_time/60:.1f} min)")

# ============================================================
# 评估
# ============================================================
print(f"\n评估中...")
pred_val = model.predict(X_val).astype(np.float64)
r2 = weighted_r2(y_val, pred_val, w_val)
print(f"  Weighted R² = {r2:.6f}")

# ============================================================
# 保存
# ============================================================
model_path = MODELS_DIR / "tabm_full_model.pth"
torch.save(model.state_dict(), model_path)
print(f"  模型: {model_path}")

pred_path = MODELS_DIR / "tabm_val_preds.npy"
np.save(pred_path, pred_val)

results = {
    "model": "TabM (K=16, all 6 GPUs)",
    "val_r2": float(r2),
    "train_time_s": train_time,
    "n_features": X_train.shape[1],
    "n_train": len(y_train),
    "n_val": len(y_val),
    "gpus": n_gpus,
}
with open(MODELS_DIR / "tabm_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"  Results: {MODELS_DIR / 'tabm_results.json'}")
print(f"\nDONE! TabM R² = {r2:.6f}")
