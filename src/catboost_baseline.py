"""
CatBoost 基线
=============
- 采样 ~5M 行训练（CPU 全量太慢）
- 原生分类特征 (feature_09/10/11)
- ordered boosting + 加权 MSE
"""

import polars as pl
import numpy as np
from catboost import CatBoostRegressor, Pool
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train_processed.parquet"
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"

# 特征
FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]
CAT_FEATURES = ["feature_09", "feature_10", "feature_11"]
CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in CAT_FEATURES]
ALL_FEAT_COLS = CONTINUOUS_COLS + CAT_FEATURES + LAG_COLS

TRAIN_END_DATE = 1400
SAMPLE_RATE = 7  # 1/7 ≈ 5.2M rows


def weighted_r2(y_true, y_pred, w):
    num = np.sum(w * (y_true - y_pred) ** 2)
    den = np.sum(w * y_true ** 2)
    return 1 - num / den


class WeightedR2Metric:
    """自定义 Weighted R² 用于 CatBoost eval"""
    def get_final_error(self, error, weight):
        return 1 - error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        approx = np.array(approxes[0])
        t = np.array(target)
        w = np.array(weight) if weight is not None else np.ones_like(t)
        num = np.sum(w * (t - approx) ** 2)
        den = np.sum(w * t ** 2)
        r2 = 1 - num / (den + 1e-38)
        return r2, 1


def main():
    t0 = time.time()
    print("=" * 60)
    print("CatBoost 基线 (采样 ~5M 行)")
    print("=" * 60)

    # ---- 1. 加载采样数据 ----
    print("\n1. 采样训练数据 (1/7)...")
    train_lazy = pl.scan_parquet(TRAIN_PATH).filter(pl.col('date_id') <= TRAIN_END_DATE)
    train_df = train_lazy.collect()
    train_df = train_df.filter(pl.int_range(0, pl.len()) % SAMPLE_RATE == 0)
    print(f"   训练集: {train_df.height:,} rows × {len(ALL_FEAT_COLS)} features")

    # 提取特征 / target / weight
    X_train = train_df[ALL_FEAT_COLS].to_pandas()
    y_train = train_df[TARGET_COL].to_numpy().astype(np.float64)
    w_train = train_df[WEIGHT_COL].to_numpy().astype(np.float64)

    # CatBoost 的 categorical feature indices
    cat_indices = [ALL_FEAT_COLS.index(c) for c in CAT_FEATURES]
    print(f"   分类特征索引: {cat_indices} ({CAT_FEATURES})")
    del train_df

    # ---- 2. 加载验证集 ----
    print("\n2. 加载验证集...")
    val_lazy = pl.scan_parquet(TRAIN_PATH).filter(pl.col('date_id') > TRAIN_END_DATE)
    val_df = val_lazy.collect()
    print(f"   验证集: {val_df.height:,} rows")

    X_val = val_df[ALL_FEAT_COLS].to_pandas()
    y_val = val_df[TARGET_COL].to_numpy().astype(np.float64)
    w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float64)
    val_sids = val_df['symbol_id'].to_numpy()
    del val_df

    # ---- 3. 训练 CatBoost ----
    print("\n3. 训练 CatBoost...")
    train_pool = Pool(X_train, y_train, weight=w_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_indices)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        random_strength=1,
        bagging_temperature=0.5,
        od_type='Iter',
        od_wait=100,
        loss_function='RMSEWithUncertainty',  # 稳定
        eval_metric='RMSE',
        random_seed=42,
        verbose=200,
        thread_count=8,
        allow_writing_files=False,
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose_eval=200,
    )

    # ---- 4. 评估 ----
    print(f"\n4. 评估...")
    y_pred = model.predict(X_val)

    val_r2 = weighted_r2(y_val, y_pred, w_val)
    print(f"   验证 R² = {val_r2:.6f}")

    # 朴素基线对比
    naive_pred = np.zeros_like(y_val)
    naive_r2 = weighted_r2(y_val, naive_pred, w_val)
    print(f"   预测全0 R² = {naive_r2:.6f}")

    # 各 symbol
    print(f"\n   各 symbol:")
    for sid in sorted(set(int(s) for s in val_sids)):
        mask = val_sids == sid
        if mask.sum() > 1000:
            r2_s = weighted_r2(y_val[mask], y_pred[mask], w_val[mask])
            naive_s = weighted_r2(y_val[mask], np.zeros(mask.sum()), w_val[mask])
            delta = r2_s - naive_s
            bar = "+" * max(0, int(delta * 500)) if delta > 0 else ""
            print(f"    symbol_{sid:02d}: R²={r2_s:.6f} (naive={naive_s:.6f}, Δ={delta:+.5f}) {bar}")

    # 特征重要性
    print(f"\n   Top 15 特征重要性:")
    importances = model.get_feature_importance()
    feat_imp = sorted(zip(ALL_FEAT_COLS, importances), key=lambda x: -x[1])
    for name, imp in feat_imp[:15]:
        print(f"    {name:25s}: {imp:.4f}")

    t = time.time() - t0
    print(f"\n总耗时: {t:.1f}s ({t/60:.1f} min)")
    print(f"CatBoost R² = {val_r2:.6f} (vs Ridge = 0.003993)")


if __name__ == "__main__":
    main()
