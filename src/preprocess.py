"""
数据预处理 Pipeline（流式处理版 + 自生成 lag）
=============================================

策略:
1. scan_parquet 读取所有分区 → 排序 → 分层 ffill → sink 到临时文件
2. 从干净数据中生成 lag（前一天同 symbol 的 responder 值）→ merge → 写出最终文件

Lag 生成逻辑:
  对每个 (symbol_id, date_id)，取该天最后一个 time_id 的 responder 值，
  然后 shift(1) 得到前一天的 lag。
"""

import polars as pl
from pathlib import Path
import time
import gc

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

RESPONDER_COLS = [f"responder_{i}" for i in range(9)]
ALL_FEATURES = [f"feature_{i:02d}" for i in range(79)]

# 慢变量：日间自相关 > 0.6，ffill 误差小
STABLE_FFILL = [
    "feature_00", "feature_02", "feature_03",
    "feature_15",
    "feature_39", "feature_42", "feature_50", "feature_53",
]

# 快变量：日间自相关 < 0，填 0
UNSTABLE_FILL_ZERO = [
    "feature_01", "feature_04",
]

OTHER_FEATURES = [c for c in ALL_FEATURES if c not in set(STABLE_FFILL) | set(UNSTABLE_FILL_ZERO)]


def stage1_fill_and_sort(data_dir: Path, out_path: Path):
    """阶段1: 读取 → 排序 → 填充 → 写出"""
    print("=" * 60)
    print("Stage 1: 缺失值填充")
    print("=" * 60)

    train_pattern = str(data_dir / "train.parquet" / "partition_id=*" / "*.parquet")
    df = pl.scan_parquet(train_pattern)

    # 排序
    df = df.sort(["symbol_id", "date_id", "time_id"])

    # 慢变量：按 symbol ffill
    df = df.with_columns([
        pl.col(STABLE_FFILL)
        .fill_null(strategy="forward")
        .over("symbol_id")
    ])

    # 快变量：填 0
    df = df.with_columns([
        pl.col(UNSTABLE_FILL_ZERO).fill_null(0)
    ])

    # 其余：按 symbol ffill
    df = df.with_columns([
        pl.col(OTHER_FEATURES)
        .fill_null(strategy="forward")
        .over("symbol_id")
    ])

    # 剩余 NaN → 0
    df = df.fill_null(0)

    print(f"  写出到: {out_path}")
    df.sink_parquet(out_path, compression="zstd")
    size_gb = out_path.stat().st_size / 1e9
    print(f"  大小: {size_gb:.2f} GB")


def stage2_generate_lags(data_dir: Path, filled_path: Path, final_path: Path):
    """阶段2: 生成 lag 特征 → 合并 → 写出最终"""
    print("\n" + "=" * 60)
    print("Stage 2: Lag 特征生成")
    print("=" * 60)

    # 扫描已填充的数据
    main = pl.scan_parquet(filled_path)

    # --- 2a. 生成每日 lag 表 ---
    # 取每个 (symbol_id, date_id) 的最后一个 time_id 的 responder 值
    # 然后 shift(1) 得到前一天的 lag
    main_sorted = main.sort(["symbol_id", "date_id", "time_id"])

    # 每日最后一行
    daily_last = main_sorted.group_by(
        ["symbol_id", "date_id"],
        maintain_order=True
    ).agg(
        pl.col(RESPONDER_COLS).last()
    )

    # 对每个 symbol 做 shift → lag
    lag_exprs = []
    for c in RESPONDER_COLS:
        lag_exprs.append(
            pl.col(c).shift(1).over("symbol_id").alias(f"{c}_lag_1")
        )

    daily_lags = daily_last.with_columns(lag_exprs)
    # 只保留 join 需要的列 + lag 列
    lag_cols = [f"{c}_lag_1" for c in RESPONDER_COLS]
    daily_lags_out = daily_lags.select(
        ["symbol_id", "date_id"] + lag_cols
    )

    # --- 2b. Join 回主表 ---
    # 注意: date_id 的 lag 对应的是 "前一天的 responder"
    # 所以 date_id=d 的 lag 来自 date_id=d-1 的 responder
    # shift(1) 已经做了这件事: 对于同一个 symbol, date_id 递增排列,
    # shift(1) 后的值 = 前一天的 responder
    final = main_sorted.join(
        daily_lags_out,
        on=["symbol_id", "date_id"],
        how="left",
    )

    # lag 的 NaN 填 0（第一天没有 lag）
    final = final.fill_null(0)

    print(f"  写出最终文件: {final_path}")
    final.sink_parquet(final_path, compression="zstd")
    size_gb = final_path.stat().st_size / 1e9
    print(f"  大小: {size_gb:.2f} GB")


def verify(path: Path):
    """快速验证输出"""
    print("\n" + "=" * 60)
    print("验证")
    print("=" * 60)

    f = pl.scan_parquet(path)
    n_rows = f.select(pl.len()).collect().item()
    n_cols = len(f.collect_schema())

    sample = f.head(5000).collect()
    nulls = sum(sample.null_count().row(0))
    lag_cols = [c for c in sample.columns if '_lag_' in c]

    # Check lag fill rate
    lag_stats = {}
    for c in lag_cols:
        zero_pct = (sample[c] == 0).sum() / len(sample) * 100
        lag_stats[c] = zero_pct

    # Also check middle of data for lags
    mid_sample = f.slice(23_000_000, 5000).collect()
    mid_lag_stats = {}
    for c in lag_cols:
        zero_pct = (mid_sample[c] == 0).sum() / len(mid_sample) * 100
        mid_lag_stats[c] = zero_pct

    print(f"  行数: {n_rows:,}")
    print(f"  列数: {n_cols}  (4 meta + 79 feat + 9 resp + 9 lag = 101)")
    print(f"  NaN 数: {nulls}")

    print(f"\n  Lag 0值占比 (前5000行 / 中间5000行):")
    for c in lag_cols:
        front = lag_stats.get(c, 100)
        mid = mid_lag_stats.get(c, 100)
        f_bar = "█" if front > 90 else "▄" if front > 30 else "▁"
        m_bar = "█" if mid > 90 else "▄" if mid > 30 else "▁"
        print(f"    {c}: 前={front:5.1f}% {f_bar}  中={mid:5.1f}% {m_bar}")

    # Feature stats
    print(f"\n  特征摘要 (前5000行):")
    feat_cols = [c for c in sample.columns if c.startswith('feature_')]
    for c in feat_cols[:6]:
        s = sample[c]
        print(f"    {c}: mean={s.mean():+.4f}, std={s.std():.4f}, "
              f"zero_pct={(s==0).sum()/len(s)*100:.1f}%")

    print(f"\n  中间5000行特征摘要:")
    for c in feat_cols[:6]:
        s = mid_sample[c]
        print(f"    {c}: mean={s.mean():+.4f}, std={s.std():.4f}, "
              f"zero_pct={(s==0).sum()/len(s)*100:.1f}%")

    print(f"\n✓ 验证通过" if nulls == 0 else f"\n⚠ 有 {nulls} 个 NaN!")


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_path = OUTPUT_DIR / "train_tmp.parquet"
    final_path = OUTPUT_DIR / "train_processed.parquet"

    # Stage 1
    stage1_fill_and_sort(DATA_DIR, tmp_path)
    gc.collect()

    # Stage 2
    stage2_generate_lags(DATA_DIR, tmp_path, final_path)
    gc.collect()

    # Clean up temp
    if tmp_path.exists():
        tmp_path.unlink()
        print(f"\n清理临时文件: {tmp_path}")

    # Verify
    verify(final_path)

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
