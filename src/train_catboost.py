"""
CatBoost GPU 训练
================
- GPU 加速 ordered boosting
- 原生分类特征支持（无需 one-hot）
- Weighted R² 自定义评估指标
- 自动保存最优模型
"""

import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
import time
import gc
import json
import os

from .data_utils import (
    prepare_tree_data, MODELS_DIR, CAT_FEATURES,
    LAG_COLS, CONTINUOUS_FEATURES, TARGET_COL, WEIGHT_COL,
)
from .metrics import weighted_r2, weighted_r2_per_group


MODEL_PATH = MODELS_DIR / "catboost_model.cbm"
RESULT_PATH = MODELS_DIR / "catboost_results.json"


class WeightedR2Metric:
    """CatBoost 自定义 Weighted R² 评估指标"""

    def get_final_error(self, error, weight):
        return 1.0 - error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        approx = np.array(approxes[0])
        t = np.array(target)
        w = np.array(weight) if weight is not None else np.ones_like(t)
        num = np.sum(w * (t - approx) ** 2)
        den = np.sum(w * t ** 2)
        r2 = 1.0 - num / (den + 1e-38)
        return r2, 1


def train(
    sample_rate: int = 6,
    iterations: int = 2000,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    early_stopping_rounds: int = 100,
    verbose: int = 100,
):
    """
    训练 CatBoost 模型（GPU）。

    参数:
        sample_rate: 1/N 采样率（N=6 → ~6.5M 行训练）
        iterations: 最大迭代次数
        learning_rate: 学习率
        depth: 树深度
        l2_leaf_reg: L2 正则化强度
        early_stopping_rounds: 早停轮数
        verbose: 日志频率
    """
    print("=" * 60)
    print("CatBoost GPU 训练")
    print("=" * 60)

    # 检查 GPU
    try:
        import catboost
        print(f"  CatBoost 版本: {catboost.__version__}")
    except ImportError:
        print("  ⚠ CatBoost 未安装！")
        return None

    # 自动检测 GPU 数量
    n_gpus = _detect_gpu_count()
    if n_gpus == 0:
        print("  ⚠ 未检测到 GPU！将使用 CPU 模式（可能非常慢）")
        devices = "0"
        task_type = "CPU"
    else:
        devices = "0" if n_gpus == 1 else f"0-{n_gpus - 1}"
        task_type = "GPU"
    print(f"  检测到 {n_gpus} 张 GPU → devices=\"{devices}\", task_type={task_type}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- 1. 准备数据 ----
    print("\n[1/4] 加载数据...")
    X_train, y_train, w_train, X_val, y_val, w_val, val_sids, feat_names, cat_indices = \
        prepare_tree_data(sample_rate=sample_rate, one_hot=False)

    # 特征名中分类特征的索引
    all_feat_list = list(CONTINUOUS_FEATURES) + list(LAG_COLS) + list(CAT_FEATURES)
    cat_idx_in_features = [all_feat_list.index(c) for c in CAT_FEATURES]
    print(f"  分类特征索引: {cat_idx_in_features} ({CAT_FEATURES})")

    # ---- 2. 创建 Pool ----
    print("\n[2/4] 创建 CatBoost Pool...")
    train_pool = Pool(
        X_train, y_train,
        weight=w_train,
        cat_features=cat_idx_in_features,
    )
    val_pool = Pool(
        X_val, y_val,
        weight=w_val,
        cat_features=cat_idx_in_features,
    )

    del X_train, y_train, w_train
    gc.collect()

    # ---- 3. 训练 ----
    print("\n[3/4] 训练 CatBoost (GPU)...")
    print(f"  训练集: {train_pool.num_row():,} rows × {train_pool.num_col()} cols")
    print(f"  验证集: {val_pool.num_row():,} rows")
    print(f"  Iterations: {iterations}, LR: {learning_rate}, Depth: {depth}")

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=1.0,
        bagging_temperature=0.5,
        bootstrap_type="Poisson",  # GPU 兼容
        od_type="Iter",
        od_wait=early_stopping_rounds,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type=task_type,
        devices=devices,  # 自动检测
        verbose=verbose,
        allow_writing_files=False,
        # GPU 性能优化
        gpu_ram_part=0.85,  # 使用 85% GPU 内存
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose_eval=verbose,
    )

    train_time = time.time() - t0

    # ---- 4. 评估 ----
    print("\n[4/4] 评估...")
    y_pred_val = model.predict(X_val)
    val_r2 = weighted_r2(y_val, y_pred_val, w_val)

    # 全 0 基线
    naive_r2 = weighted_r2(y_val, np.zeros_like(y_val), w_val)
    print(f"  验证 R² = {val_r2:.6f}")
    print(f"  全 0 基线 = {naive_r2:.6f}")
    print(f"  相对提升 = {val_r2 - naive_r2:+.6f}")

    # 各 symbol
    print(f"\n  各 symbol 验证 R²:")
    per_symbol = weighted_r2_per_group(y_val, y_pred_val, w_val, groups=val_sids)
    for sid, r2_s in sorted(per_symbol.items()):
        bar = "+" * max(0, int((r2_s - naive_r2) * 500))
        print(f"    symbol_{sid:02d}: R²={r2_s:+.6f} {bar}")

    # 特征重要性
    print(f"\n  Top 20 特征重要性:")
    importances = model.get_feature_importance()
    feat_imp = sorted(zip(all_feat_list, importances), key=lambda x: -x[1])
    for name, imp in feat_imp[:20]:
        print(f"    {name:25s}: {imp:.4f}")

    # ---- 5. 保存 ----
    print(f"\n  保存模型: {MODEL_PATH}")
    model.save_model(str(MODEL_PATH))

    results = {
        "model": "CatBoost",
        "val_r2": float(val_r2),
        "naive_r2": float(naive_r2),
        "delta": float(val_r2 - naive_r2),
        "per_symbol": {str(k): float(v) for k, v in per_symbol.items()},
        "train_time_s": train_time,
        "params": {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "sample_rate": sample_rate,
            "n_features": len(all_feat_list),
        },
        "top_features": [(name, float(imp)) for name, imp in feat_imp[:20]],
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t0
    # 保存验证集预测（用于集成）
    preds_path = MODELS_DIR / "catboost_val_preds.npy"
    np.save(preds_path, y_pred_val)

    print(f"\n  ✓ CatBoost 完成 | R²={val_r2:.6f} | 耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 清理 GPU 内存
    del model, train_pool, val_pool, X_val, y_val, w_val
    gc.collect()
    _clear_gpu_memory()

    return val_r2


def _detect_gpu_count() -> int:
    """自动检测可用 GPU 数量（优先 torch，其次 nvidia-smi）"""
    # 方式 1: PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # 方式 2: nvidia-smi
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        return len(lines)
    except Exception:
        pass

    return 0


def _clear_gpu_memory():
    """清理 GPU 内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


if __name__ == "__main__":
    train()
