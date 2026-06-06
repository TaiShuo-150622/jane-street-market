"""
Ridge 基线（全流式处理 + 闭式解）
=================================
- 流式计算特征统计量 (mean/std)
- 分块计算 XᵀWX 和 XᵀWy
- 闭式解: β = (XᵀWX + λI)⁻¹ XᵀWy
- 全程内存可控，不加载全量数据
"""

import polars as pl
import numpy as np
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train_processed.parquet"
TARGET_COL = "responder_6"
WEIGHT_COL = "weight"

# 分类特征 + 预扫描得到的全部唯一值
CAT_FEATURES = {
    "feature_09": [2, 4, 9, 11, 12, 14, 15, 25, 26, 30, 34, 42, 44, 46, 49, 50, 57, 64, 68, 70, 81, 82],
    "feature_10": [1, 2, 3, 4, 5, 6, 7, 10, 12],
    "feature_11": [9, 11, 13, 16, 24, 25, 34, 40, 48, 50, 59, 62, 63, 66, 76, 150, 158, 159, 171, 195, 214, 230, 261, 297, 336, 376, 388, 410, 522, 534, 539],
}
# 预计算 one-hot 列名
OH_COLUMNS = []
for cat, vals in CAT_FEATURES.items():
    for v in vals:
        OH_COLUMNS.append(f"{cat}_{v}")

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]
# ⚠️ 不用当前时刻的 responder (数据泄露)! 只用 lag (前一天的)
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]
CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in CAT_FEATURES]
BASE_FEAT_NAMES = list(CONTINUOUS_COLS) + LAG_COLS
ALL_FEAT_NAMES = BASE_FEAT_NAMES + OH_COLUMNS

TRAIN_END_DATE = 1400
CHUNK_SIZE = 2_000_000


def get_feature_names(df_schema: list) -> list:
    """构建特征名列表（含 one-hot 展开后的列名）"""
    # 基准: 连续特征 + responder + lag
    names = list(CONTINUOUS_COLS) + RESPONDER_COLS + LAG_COLS
    return names


def stream_compute_stats(lazy_df, feat_names):
    """流式计算每个特征的 mean 和 std"""
    exprs = []
    for c in feat_names:
        exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        exprs.append(pl.col(c).std().alias(f"{c}_std"))

    stats_df = lazy_df.select(exprs).collect()
    stats = {}
    for c in feat_names:
        m = stats_df[f"{c}_mean"][0]
        s = stats_df[f"{c}_std"][0]
        if s is None or s == 0 or np.isnan(s):
            s = 1.0
        if m is None or np.isnan(m):
            m = 0.0
        stats[c] = {'mean': float(m), 'std': float(s)}
    return stats


def process_chunk(df_chunk, stats):
    """处理一个数据块: 固定 one-hot + 标准化 → X, y, w"""
    n_rows = df_chunk.height

    # 1. 连续特征 + responder + lag (已标准化)
    X_parts = []
    for c in BASE_FEAT_NAMES:
        vals = df_chunk[c].to_numpy().astype(np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        if c in stats:
            m, s = stats[c]['mean'], stats[c]['std']
            vals = (vals - m) / s
        X_parts.append(vals.reshape(-1, 1))

    # 2. One-hot: 全量固定列, 检查匹配
    for cat_col, cat_vals in CAT_FEATURES.items():
        col_vals = df_chunk[cat_col].to_numpy()
        for uv in cat_vals:
            oh = (col_vals == uv).astype(np.float64)
            X_parts.append(oh.reshape(-1, 1))

    X = np.hstack(X_parts)
    y = df_chunk[TARGET_COL].to_numpy().astype(np.float64)
    y = np.nan_to_num(y, nan=0.0)
    w = df_chunk[WEIGHT_COL].to_numpy().astype(np.float64)
    w = np.nan_to_num(w, nan=0.0)

    return X, y, w


def weighted_r2(y_true, y_pred, w):
    num = np.sum(w * (y_true - y_pred) ** 2)
    den = np.sum(w * y_true ** 2)
    return 1 - num / den


def main():
    t0 = time.time()
    print("=" * 60)
    print("Ridge 基线 (全流式 + 闭式解)")
    print("=" * 60)

    # ---- 1. 流式计算训练集统计量 ----
    print("\n1. 流式计算特征统计量...")
    t1 = time.time()
    train_lazy = pl.scan_parquet(TRAIN_PATH).filter(pl.col('date_id') <= TRAIN_END_DATE)
    stats = stream_compute_stats(train_lazy, BASE_FEAT_NAMES)
    print(f"   耗时: {time.time()-t1:.1f}s")

    # ---- 2. 分块计算 XᵀWX 和 XᵀWy ----
    print("\n2. 分块计算 XᵀWX, XᵀWy...")
    p = len(ALL_FEAT_NAMES)
    print(f"   特征总数: {p}  (连续={len(CONTINUOUS_COLS)}, "
          f"lag={len(LAG_COLS)}, one-hot={len(OH_COLUMNS)})")

    # Initialize accumulators
    xtx = np.zeros((p, p), dtype=np.float64)
    xty = np.zeros(p, dtype=np.float64)
    wy2 = 0.0
    total_rows = 0

    def add_chunk(Xc, yc, wc):
        nonlocal xtx, xty, wy2, total_rows
        n = Xc.shape[0]
        Xw = Xc * wc[:, np.newaxis]
        xtx += Xc.T @ Xw
        xty += Xw.T @ yc
        wy2 += float(np.dot(wc, yc * yc))
        total_rows += n

    n_est = train_lazy.select(pl.len()).collect().item()
    n_chunks = (n_est + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"   估计 {n_chunks} 个 chunk, 共 {n_est:,} 行")

    for chunk_i in range(n_chunks):
        offset = chunk_i * CHUNK_SIZE
        chunk = train_lazy.slice(offset, CHUNK_SIZE).collect()
        if chunk.height == 0:
            break
        Xc, yc, wc = process_chunk(chunk, stats)
        add_chunk(Xc, yc, wc)
        if chunk_i % 5 == 0:
            print(f"   chunk {chunk_i}/{n_chunks}: {total_rows:,} rows")

    print(f"   完成: {total_rows:,} rows")
    print(f"   矩阵大小: XᵀWX {p}×{p}, XᵀWy {p}×1")

    # ---- 3. 加载验证集 ----
    print("\n3. 加载验证集...")
    val_lazy = pl.scan_parquet(TRAIN_PATH).filter(pl.col('date_id') > TRAIN_END_DATE)
    xs, ys, ws, sids = [], [], [], []
    n_val_est = val_lazy.select(pl.len()).collect().item()
    n_val_chunks = (n_val_est + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_i in range(n_val_chunks):
        chunk = val_lazy.slice(chunk_i * CHUNK_SIZE, CHUNK_SIZE).collect()
        if chunk.height == 0:
            break
        Xc, yc, wc = process_chunk(chunk, stats)
        xs.append(Xc); ys.append(yc); ws.append(wc)
        sids.append(chunk['symbol_id'].to_numpy())
    X_val = np.concatenate(xs); y_val = np.concatenate(ys)
    w_val = np.concatenate(ws); val_sids = np.concatenate(sids)
    print(f"   验证集: {len(y_val):,} rows")

    # ---- 4. Ridge 闭式解 + λ 搜索 ----
    print(f"\n4. λ 搜索...")
    lambdas = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    best_lam, best_r2 = None, -np.inf

    for lam in lambdas:
        reg_xtx = xtx + lam * np.eye(p)
        try:
            beta = np.linalg.solve(reg_xtx, xty)
        except np.linalg.LinAlgError:
            print(f"  λ={lam:8.1f}: 矩阵奇异，跳过")
            continue

        y_pred = X_val @ beta
        val_r2 = weighted_r2(y_val, y_pred, w_val)
        train_r2 = 1 - (beta @ xtx @ beta - 2 * beta @ xty + wy2) / wy2

        marker = " ←" if val_r2 > best_r2 else ""
        print(f"  λ={lam:8.1f}  train_R²={train_r2:.6f}  val_R²={val_r2:.6f}{marker}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_lam = lam

    # ---- 5. 最终模型 ----
    print(f"\n5. 最佳: λ={best_lam}, val_R²={best_r2:.6f}")
    beta = np.linalg.solve(xtx + best_lam * np.eye(p), xty)
    y_pred_final = X_val @ beta

    # 特征重要性
    importances = np.abs(beta)
    top_idx = np.argsort(importances)[-20:][::-1]
    print(f"\n  Top 20 特征:")
    for idx in top_idx:
        print(f"    {ALL_FEAT_NAMES[idx]:28s}: |β|={importances[idx]:.6f}")

    # 各 symbol
    print(f"\n  各 symbol 验证 R²:")
    for sid in sorted(set(int(s) for s in val_sids)):
        mask = val_sids == sid
        if mask.sum() > 1000:
            r2_s = weighted_r2(y_val[mask], y_pred_final[mask], w_val[mask])
            print(f"    symbol_{sid:02d}: R²={r2_s:.6f} (n={mask.sum():,})")

    t = time.time() - t0
    print(f"\n总耗时: {t:.1f}s ({t/60:.1f} min) | 基线 R² = {best_r2:.6f}")


if __name__ == "__main__":
    main()
