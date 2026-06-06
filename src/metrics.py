"""
统一评估指标：Weighted R²
=======================
Kaggle Jane Street 比赛使用的评估指标。
R²_w = 1 - Σ w_i (y_i - ŷ_i)² / Σ w_i y_i²
"""

import numpy as np


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """
    Weighted R² 指标。

    Args:
        y_true: 真实值, shape (n,)
        y_pred: 预测值, shape (n,)
        weight: 样本权重, shape (n,)

    Returns:
        float: R² 值，越大越好。理论上界 1.0，无下界。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    denominator = np.sum(weight * y_true ** 2)

    if denominator < 1e-38:
        return 0.0

    return float(1.0 - numerator / denominator)


def weighted_r2_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """计算每个 group 的 weighted R²。"""
    scores = {}
    for g in sorted(set(int(g) for g in groups)):
        mask = groups == g
        if mask.sum() > 100:
            scores[g] = weighted_r2(y_true[mask], y_pred[mask], weight[mask])
    return scores
