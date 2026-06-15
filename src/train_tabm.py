"""
TabM 训练 — 使用官方 tabm 包
=============================
pip install tabm
替换手写 tanm_reference.py，用 Yandex 官方实现
"""
import numpy as np
import polars as pl
import json, time, gc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.data_utils import MODELS_DIR, TRAIN_PATH, TARGET_COL, WEIGHT_COL, TRAIN_END_DATE
from src.metrics import weighted_r2

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79) if i not in (9, 10, 11)]
CAT_FEATURE_COLS = ["feature_09", "feature_10", "feature_11"]
RESPONDER_COLS = [f"responder_{i}" for i in range(9) if i != 6]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]


def load_data():
    """加载预处理后的数据"""
    print("  加载数据...")
    df = pl.read_parquet(TRAIN_PATH)

    # 分割 train/val（按 date_id 时间切分）
    train_df = df.filter(pl.col("date_id") <= TRAIN_END_DATE)
    val_df = df.filter(pl.col("date_id") > TRAIN_END_DATE)

    # 特征列
    all_features = FEATURE_COLS + CAT_FEATURE_COLS + RESPONDER_COLS + LAG_COLS
    # 只选存在的列
    avail = [c for c in all_features if c in train_df.columns]

    X_train = train_df.select(avail).to_numpy().astype(np.float32)
    y_train = train_df[TARGET_COL].to_numpy().astype(np.float32).ravel()
    w_train = train_df[WEIGHT_COL].to_numpy().astype(np.float32).ravel()

    X_val = val_df.select(avail).to_numpy().astype(np.float32)
    y_val = val_df[TARGET_COL].to_numpy().astype(np.float32).ravel()
    w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float32).ravel()

    # NaN → 0
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Features: {X_train.shape[1]}")
    return X_train, y_train, w_train, X_val, y_val, w_val


def train():
    """训练 TabM 回归模型"""
    import torch
    from tabm import TabMRegressor

    n_gpus = torch.cuda.device_count()
    device = "cuda:0" if n_gpus > 0 else "cpu"
    print(f"  设备: {device}, GPU 数: {n_gpus}")

    # 加载数据
    X_train, y_train, w_train, X_val, y_val, w_val = load_data()

    # TabM 参数字典
    model = TabMRegressor(
        device=device,
        n_estimators=8,           # K=8 个子模型
        random_state=42,
    )

    print(f"  训练 TabM (K=8, device={device})...")
    t0 = time.time()

    # TabM 的 fit 自动做内部验证
    model.fit(X_train, y_train)

    train_time = time.time() - t0
    print(f"  训练完成: {train_time:.0f}s")

    # 预测 + 评估
    print("  验证集预测...")
    pred_val = model.predict(X_val).astype(np.float64)
    r2 = weighted_r2(y_val, pred_val, w_val)
    print(f"  验证集 Weighted R² = {r2:.6f}")

    # 保存
    model_path = MODELS_DIR / "tabm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  模型: {model_path}")

    pred_path = MODELS_DIR / "tabm_val_preds.npy"
    np.save(pred_path, pred_val)
    print(f"  验证预测: {pred_path}")

    result = {"model": "TabM (official)", "val_r2": float(r2), "train_time_s": train_time,
              "n_features": X_train.shape[1], "n_train": len(y_train)}
    with open(MODELS_DIR / "tabm_results.json", "w") as f:
        json.dump(result, f, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return r2


if __name__ == "__main__":
    train()
