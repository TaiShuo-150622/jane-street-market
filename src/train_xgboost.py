"""
XGBoost GPU 训练
================
- GPU histogram 加速 (gpu_hist)
- One-hot 编码分类特征
- Weighted MSE 训练
- 早停 + 最优模型保存
"""

import numpy as np
import xgboost as xgb
from pathlib import Path
import time
import gc
import json

from .data_utils import prepare_tree_data, MODELS_DIR, CAT_FEATURES
from .metrics import weighted_r2, weighted_r2_per_group


MODEL_PATH = MODELS_DIR / "xgboost_model.json"
RESULT_PATH = MODELS_DIR / "xgboost_results.json"


def train(
    sample_rate: int = 6,
    n_estimators: int = 1000,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 1.0,
    reg_lambda: float = 5.0,
    early_stopping_rounds: int = 100,
    verbose: int = 50,
):
    """
    训练 XGBoost 模型（GPU）。

    参数:
        sample_rate: 1/N 采样率
        n_estimators: 最大树数
        learning_rate: 学习率
        max_depth: 树深度
        subsample: 行采样率
        colsample_bytree: 列采样率
        reg_alpha: L1 正则化
        reg_lambda: L2 正则化
        early_stopping_rounds: 早停轮数
        verbose: 日志频率
    """
    print("=" * 60)
    print("XGBoost GPU 训练")
    print("=" * 60)

    # 检查版本
    print(f"  XGBoost 版本: {xgb.__version__}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- 1. 准备数据 ----
    print("\n[1/3] 加载数据...")
    X_train, y_train, w_train, X_val, y_val, w_val, val_sids, feat_names, _ = \
        prepare_tree_data(sample_rate=sample_rate, one_hot=True)

    print(f"  训练集: {X_train.shape[0]:,} rows × {X_train.shape[1]} cols")
    print(f"  验证集: {X_val.shape[0]:,} rows × {X_val.shape[1]} cols")

    # ---- 2. 创建 DMatrix ----
    print("\n[2/3] 创建 DMatrix + 训练...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    # 释放原始数据（DMatrix 持有副本）
    del X_train, y_train, w_train
    gc.collect()

    params = {
        "objective": "reg:squarederror",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "tree_method": "gpu_hist",
        "device": "cuda",  # 自动使用所有可见 GPU
        "random_state": 42,
        "verbosity": 1,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    evals_result: dict = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )

    train_time = time.time() - t0

    # ---- 3. 评估 ----
    print("\n[3/3] 评估...")

    # 最佳迭代
    best_iteration = model.best_iteration
    print(f"  最佳迭代: {best_iteration}")

    y_pred_val = model.predict(dval)
    val_r2 = weighted_r2(y_val, y_pred_val, w_val)
    naive_r2 = weighted_r2(y_val, np.zeros_like(y_val), w_val)

    print(f"  验证 R² = {val_r2:.6f}")
    print(f"  全 0 基线 = {naive_r2:.6f}")
    print(f"  相对提升 = {val_r2 - naive_r2:+.6f}")

    # 各 symbol
    per_symbol = weighted_r2_per_group(y_val, y_pred_val, w_val, groups=val_sids)
    print(f"\n  各 symbol 验证 R²:")
    for sid, r2_s in sorted(per_symbol.items()):
        bar = "+" * max(0, int((r2_s - naive_r2) * 500))
        print(f"    symbol_{sid:02d}: R²={r2_s:+.6f} {bar}")

    # 特征重要性
    print(f"\n  Top 20 特征重要性 (gain):")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:20]
    for name, imp in sorted_imp:
        # name 是 f0, f1, ... 格式，映射回特征名
        try:
            idx = int(name[1:])
            real_name = feat_names[idx] if idx < len(feat_names) else name
        except (ValueError, IndexError):
            real_name = name
        print(f"    {real_name:25s}: {imp:.4f}")

    # ---- 保存 ----
    print(f"\n  保存模型: {MODEL_PATH}")
    model.save_model(str(MODEL_PATH))

    results = {
        "model": "XGBoost",
        "val_r2": float(val_r2),
        "naive_r2": float(naive_r2),
        "delta": float(val_r2 - naive_r2),
        "best_iteration": int(best_iteration),
        "per_symbol": {str(k): float(v) for k, v in per_symbol.items()},
        "train_time_s": train_time,
        "params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "sample_rate": sample_rate,
            "n_features": len(feat_names),
        },
        "top_features": [(str(name), float(imp)) for name, imp in sorted_imp],
        "eval_history": {
            k: [float(x) for x in v["val"]] if "val" in v else []
            for k, v in evals_result.items()
        },
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n  ✓ XGBoost 完成 | R²={val_r2:.6f} | 耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 保存验证集预测（用于集成）
    preds_path = MODELS_DIR / "xgboost_val_preds.npy"
    np.save(preds_path, y_pred_val.astype(np.float32))

    # 清理
    del model, dtrain, dval, X_val, y_val, w_val
    gc.collect()
    _clear_gpu()

    return val_r2


def _clear_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    train()
