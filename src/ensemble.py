"""
集成权重搜索 & 推理
==================
- 网格搜索最优集成权重（最大化验证集 weighted R²）
- 支持 Ridge regression stacking (meta-learner)
- 保存集成配置
"""

import numpy as np
from pathlib import Path
import json
import time
from typing import Optional

from .metrics import weighted_r2, weighted_r2_per_group
from .data_utils import (
    _get_val_lazy, TARGET_COL, WEIGHT_COL, MODELS_DIR,
    CONTINUOUS_FEATURES, CAT_FEATURES, LAG_COLS,
)


def load_model_predictions(
    model_paths: dict[str, Path],
    model_types: dict[str, str],
    feature_set_info: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """
    加载各模型在验证集上的预测。

    参数:
        model_paths: {model_name: path_to_model_file}
        model_types: {model_name: "catboost"|"xgboost"|"mlp"}
        feature_set_info: {model_name: feature_name_list}

    返回:
        {model_name: y_pred_numpy_array}
    """
    ...


def grid_search_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    w_val: np.ndarray,
    step: float = 0.05,
) -> dict:
    """
    网格搜索最优集成权重。

    对于 N 个模型，搜索所有权重组合 w_i ∈ [0, 1]，
    Σ w_i = 1（归一化），步长 = step。

    返回:
        {"weights": {name: weight}, "val_r2": float}
    """
    ...


def stacking_ridge(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    w_val: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """
    Ridge regression stacking: 用验证集预测值作为特征，
    拟合 Ridge 得到 meta-learner 权重。

    返回:
        {"weights": {name: weight}, "intercept": float, "val_r2": float}
    """
    ...


def load_and_ensemble(
    predictions: dict[str, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    val_sids: np.ndarray,
    mode: str = "grid_search",
    grid_step: float = 0.05,
) -> dict:
    """
    主集成函数。

    参数:
        predictions: {model_name: y_pred array on validation set}
        y_val: 验证集真实值
        w_val: 验证集权重
        val_sids: 验证集 symbol_id
        mode: "grid_search" | "stacking" | "simple_average"
        grid_step: 网格搜索步长

    返回:
        结果 dict
    """
    print("=" * 60)
    print("集成权重搜索")
    print("=" * 60)

    model_names = list(predictions.keys())
    print(f"  模型: {model_names}")
    print(f"  模式: {mode}")

    # 各模型单独 R²
    print(f"\n  各模型单独验证 R²:")
    individual_scores = {}
    for name in model_names:
        r2 = weighted_r2(y_val, predictions[name], w_val)
        individual_scores[name] = r2
        print(f"    {name:20s}: R²={r2:+.6f}")

    if mode == "simple_average":
        # 简单平均
        weights = {name: 1.0 / len(model_names) for name in model_names}
        ensemble_pred = np.mean([predictions[name] for name in model_names], axis=0)
        ensemble_r2 = weighted_r2(y_val, ensemble_pred, w_val)

    elif mode == "grid_search":
        # 网格搜索
        print(f"\n  网格搜索 (step={grid_step})...")
        best_r2 = -np.inf
        best_weights = None

        n = len(model_names)
        # 生成所有权重组合（和为 1，步长 step）
        # 使用递归生成网格点
        def generate_weights(n_models, remaining, current):
            if n_models == 1:
                yield current + [round(remaining, 4)]
            else:
                i = 0.0
                while i <= remaining + 1e-8:
                    yield from generate_weights(n_models - 1, remaining - i, current + [round(i, 4)])
                    i += grid_step

        n_combos = 0
        for w_list in generate_weights(n, 1.0, []):
            n_combos += 1
            ensemble_pred = np.zeros_like(y_val)
            for i, name in enumerate(model_names):
                ensemble_pred += w_list[i] * predictions[name]

            r2 = weighted_r2(y_val, ensemble_pred, w_val)
            if r2 > best_r2:
                best_r2 = r2
                best_weights = w_list

        weights = {name: float(w) for name, w in zip(model_names, best_weights)}
        ensemble_pred = np.sum(
            [weights[name] * predictions[name] for name in model_names], axis=0
        )
        ensemble_r2 = best_r2
        print(f"  搜索组合数: {n_combos}")

    elif mode == "stacking":
        # Ridge stacking
        from sklearn.linear_model import Ridge
        print(f"\n  Ridge stacking...")

        # 构建 meta-features
        X_meta = np.column_stack([predictions[name] for name in model_names])
        meta = Ridge(alpha=1.0, fit_intercept=True, positive=False)
        meta.fit(X_meta, y_val, sample_weight=w_val)

        weights = {name: float(meta.coef_[i]) for i, name in enumerate(model_names)}
        ensemble_pred = meta.predict(X_meta)
        ensemble_r2 = weighted_r2(y_val, ensemble_pred, w_val)
        intercept = float(meta.intercept_)
        print(f"  Intercept: {intercept:.6f}")
        print(f"  Weights: {weights}")

    naive_r2 = weighted_r2(y_val, np.zeros_like(y_val), w_val)

    print(f"\n  集成结果:")
    print(f"    个体最佳 R² = {max(individual_scores.values()):.6f}")
    print(f"    集成 R²     = {ensemble_r2:.6f}")
    print(f"    全 0 基线   = {naive_r2:.6f}")
    print(f"    相对提升    = {ensemble_r2 - max(individual_scores.values()):+.6f}")

    # 各 symbol
    print(f"\n  各 symbol 集成 R²:")
    per_symbol = weighted_r2_per_group(y_val, ensemble_pred, w_val, groups=val_sids)
    for sid, r2_s in sorted(per_symbol.items()):
        # 个体最佳
        best_ind = max(weighted_r2_per_group(
            y_val[val_sids == sid],
            predictions[list(model_names)[0]][val_sids == sid],
            w_val[val_sids == sid],
            groups=val_sids[val_sids == sid],
        ).values(), default=0)
        bar = "+" * max(0, int((r2_s - naive_r2) * 500))
        print(f"    symbol_{sid:02d}: R²={r2_s:+.6f} {bar}")

    # 保存
    result_path = MODELS_DIR / "ensemble_results.json"
    results = {
        "mode": mode,
        "model_names": model_names,
        "individual_scores": {k: float(v) for k, v in individual_scores.items()},
        "weights": weights,
        "ensemble_r2": float(ensemble_r2),
        "naive_r2": float(naive_r2),
        "delta_vs_best_individual": float(ensemble_r2 - max(individual_scores.values())),
        "per_symbol": {str(k): float(v) for k, v in per_symbol.items()},
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {result_path}")

    return results


def generate_predictions(
    model_paths: dict[str, Path],
    model_types: dict[str, str],
    weights: dict[str, float],
    data_lazy,
    feature_set_info: dict[str, list[str]],
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """
    在新数据上生成集成预测（流式）。

    参数:
        model_paths: {name: path}
        model_types: {name: "catboost"|"xgboost"|"mlp"}
        weights: {name: weight}
        data_lazy: polars LazyFrame
        feature_set_info: {name: [feature columns]}
        chunk_size: 每批行数

    返回:
        y_pred: numpy array
    """
    import torch
    import catboost

    # 加载所有模型
    models = {}
    for name, path in model_paths.items():
        mtype = model_types[name]
        if mtype == "catboost":
            models[name] = catboost.CatBoostRegressor().load_model(str(path))
        elif mtype == "xgboost":
            import xgboost as xgb
            models[name] = xgb.Booster()
            models[name].load_model(str(path))
        elif mtype == "mlp":
            # Need to know the architecture parameters
            # For now, load checkpoint
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            from .train_mlp import MLP
            model = MLP(
                checkpoint["input_dim"],
                checkpoint["hidden_dims"],
                [0.1, 0.1],  # default
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[name] = model

    # 流式预测
    n_total = data_lazy.select(pl.len()).collect().item()
    all_preds = []

    for offset in range(0, n_total, chunk_size):
        chunk = data_lazy.slice(offset, chunk_size).collect()
        n = chunk.height
        if n == 0:
            break

        ensemble_chunk = np.zeros(n, dtype=np.float64)
        for name, model in models.items():
            mtype = model_types[name]
            feats = feature_set_info[name]

            if mtype == "catboost":
                X = chunk[feats].to_pandas()
                pred = model.predict(X)
                ensemble_chunk += weights[name] * pred
            elif mtype == "xgboost":
                X = chunk[feats].to_numpy().astype(np.float64)
                dmat = xgb.DMatrix(X)
                pred = model.predict(dmat)
                ensemble_chunk += weights[name] * pred
            elif mtype == "mlp":
                X = chunk[feats].to_numpy().astype(np.float32)
                X = torch.from_numpy(X)
                with torch.no_grad():
                    pred = model(X).numpy()
                ensemble_chunk += weights[name] * pred

        all_preds.append(ensemble_chunk)

        if offset % (chunk_size * 5) == 0:
            print(f"  预测进度: {offset + n:,}/{n_total:,}")

    return np.concatenate(all_preds)


if __name__ == "__main__":
    # 示例用法
    print("请在 train_all.py 中调用此模块。")
