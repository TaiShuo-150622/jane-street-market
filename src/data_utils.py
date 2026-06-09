"""
统一数据加载 & 特征工程
======================
- 流式统计量计算
- TDA 聚类 + PCA 降维（42 维特征集）
- 为不同模型准备 numpy 数组
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import gc
import time
import os

# ============================================================
# 路径 & 常量
# ============================================================
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train_processed.parquet"
MODELS_DIR = REPO_ROOT / "models"

TARGET_COL = "responder_6"
WEIGHT_COL = "weight"
TRAIN_END_DATE = 1400

# 所有 79 个 feature 列
ALL_FEATURES = [f"feature_{i:02d}" for i in range(79)]

# 分类特征
CAT_FEATURES = ["feature_09", "feature_10", "feature_11"]

# 连续特征（不含分类特征）
CONTINUOUS_FEATURES = [c for c in ALL_FEATURES if c not in CAT_FEATURES]

# Lag 特征（前一天同 symbol 的 responder 值）
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]

# Responder 列（不含 target responder_6）
RESPONDER_COLS = [f"responder_{i}" for i in range(9) if i != 6]

# ---- 全量 96 维特征集 ----
FULL_FEATURES_96 = CONTINUOUS_FEATURES + CAT_FEATURES + LAG_COLS

# ---- Regime 感知特征（torsion 分析发现的关键偏离特征）----
# 这些特征在 market regime shift 时发生明显偏移 → 模型预测摇摆
REGIME_KEY_FEATURES = ["feature_39", "feature_50", "feature_41", "feature_05", "feature_08"]
REGIME_DELTA_NAMES = [f"delta_{f}" for f in REGIME_KEY_FEATURES]
REGIME_SCORE_NAME = ["regime_score"]


def _compute_ricci_weights(
    y: np.ndarray, symbol_ids: np.ndarray, window: int = 5
) -> np.ndarray:
    """
    在每个 symbol 的时间序列上计算离散 Ricci 曲率代理，转为样本权重。

    Ricci = var(近期 window 个点) / var(远期 window 个点)
        > 1 → 发散（目标波动加剧）→ 降权
        < 1 → 收敛（目标趋于稳定）→ 提权

    返回: 权重数组，clip 到 [0.5, 2.0]
    """
    n = len(y)
    weights = np.ones(n, dtype=np.float32)
    unique_sids = np.unique(symbol_ids)

    for sid in unique_sids:
        mask = symbol_ids == sid
        idx = np.where(mask)[0]
        vals = y[idx]
        if len(vals) < 2 * window:
            continue

        ricci = np.ones(len(vals), dtype=np.float32)
        for i in range(2 * window - 1, len(vals)):
            recent = vals[i - window + 1 : i + 1]
            older = vals[i - 2 * window + 1 : i - window + 1]
            v_old = float(np.var(older))
            if v_old > 1e-12:
                ricci[i] = float(np.var(recent)) / v_old

        # 发散 → 高 Ricci → 低权重；收敛 → 低 Ricci → 高权重
        w = 1.0 / (0.5 + np.clip(ricci, 0.1, 10.0))
        w = np.clip(w, 0.5, 2.0)
        weights[idx] = w.astype(np.float32)

    return weights

# ---- TDA 42 维特征集（由 compute_tda_clusters 运行时填充）----
# 5 个 PCA 主成分 + 17 个孤立特征 + 3 个分类 + 8 个 responder + 9 个 lag
TDA_CLUSTER_FEATURES: dict[str, list[str]] = {}  # {cluster_name: [feature_list]}
TDA_ISOLATED_FEATURES: list[str] = []
TDA_PCA_WEIGHTS: dict[str, np.ndarray] = {}  # {cluster_name: pca_weight_vector}
TDA_42_FEATURES: list[str] = []


def _get_train_lazy():
    """获取训练集 LazyFrame（date_id <= TRAIN_END_DATE）"""
    return pl.scan_parquet(TRAIN_PATH).filter(pl.col("date_id") <= TRAIN_END_DATE)


def _get_val_lazy():
    """获取验证集 LazyFrame（date_id > TRAIN_END_DATE）"""
    return pl.scan_parquet(TRAIN_PATH).filter(pl.col("date_id") > TRAIN_END_DATE)


# ============================================================
# 流式统计量
# ============================================================

def compute_stats(lazy_df, feature_cols: list[str]) -> dict:
    """
    流式计算每个特征的 mean 和 std。
    返回 {col: {'mean': float, 'std': float}}
    """
    exprs = []
    for c in feature_cols:
        exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        exprs.append(pl.col(c).std(ddof=1).alias(f"{c}_std"))

    stats_df = lazy_df.select(exprs).collect()
    stats = {}
    for c in feature_cols:
        m = stats_df[f"{c}_mean"][0]
        s = stats_df[f"{c}_std"][0]
        if s is None or s == 0 or np.isnan(s):
            s = 1.0
        if m is None or np.isnan(m):
            m = 0.0
        stats[c] = {"mean": float(m), "std": float(s)}
    return stats


# ============================================================
# TDA 聚类分析（运行时计算，用于 MLP v2 的 42 维特征集）
# ============================================================

def compute_tda_clusters(
    n_sample: int = 200_000,
    corr_threshold: float = 0.65,
    min_cluster_size: int = 3,
):
    """
    在训练集样本上计算特征关联聚类，提取：
    - 聚类成员（≥3 个特征的族群）
    - 每个族群的 PCA 第一主成分权重
    - 孤立特征（不属于任何族群的特征）

    参数:
        n_sample: 用于计算相关矩阵的样本行数
        corr_threshold: |correlation| 阈值，高于此值的特征被聚在一起
        min_cluster_size: 最小族群大小

    产出（写入全局变量）:
        TDA_CLUSTER_FEATURES: {cluster_name: [feature_list]}
        TDA_ISOLATED_FEATURES: [feature_list]
        TDA_PCA_WEIGHTS: {cluster_name: np.ndarray}
        TDA_42_FEATURES: [42 维特征名列表]
    """
    global TDA_CLUSTER_FEATURES, TDA_ISOLATED_FEATURES, TDA_PCA_WEIGHTS, TDA_42_FEATURES

    print("=" * 60)
    print("TDA 聚类分析: 计算特征关联结构...")
    print("=" * 60)

    t0 = time.time()

    # 1. 采样数据
    print(f"  采样 {n_sample:,} 行用于相关矩阵计算...")
    train_lazy = _get_train_lazy()
    n_total = train_lazy.select(pl.len()).collect().item()
    step = max(1, n_total // n_sample)
    sample = train_lazy.filter(
        pl.int_range(0, pl.len()) % step == 0
    ).select(CONTINUOUS_FEATURES).collect()

    # 填充可能的 NaN → 0
    sample = sample.fill_null(0)
    X_sample = sample.to_numpy().astype(np.float64)

    print(f"  实际采样: {X_sample.shape[0]:,} rows × {X_sample.shape[1]} features")

    # 2. 计算 |correlation| 矩阵
    print("  计算 |correlation| 矩阵...")
    corr = np.corrcoef(X_sample, rowvar=False)  # shape (76, 76)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)  # 去掉自相关

    # 3. 距离矩阵 → 单链接聚类
    distance = 1.0 - abs_corr
    # 对称化 + 去除浮点误差
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    condensed_dist = squareform(distance, checks=False)
    Z = linkage(condensed_dist, method="single")

    # 4. 在 distance < (1 - corr_threshold) 处切分
    threshold = 1.0 - corr_threshold
    labels = fcluster(Z, t=threshold, criterion="distance")

    # 5. 整理族群
    cluster_map: dict[int, list[str]] = {}
    for i, feat in enumerate(CONTINUOUS_FEATURES):
        lbl = labels[i]
        cluster_map.setdefault(lbl, []).append(feat)

    clusters = {k: v for k, v in cluster_map.items() if len(v) >= min_cluster_size}
    isolated = []
    for feat in CONTINUOUS_FEATURES:
        lbl = labels[CONTINUOUS_FEATURES.index(feat)]
        if len(cluster_map[lbl]) < min_cluster_size:
            isolated.append(feat)

    # 为族群命名
    cluster_names = {}
    for i, (lbl, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        cluster_names[lbl] = f"cluster_{i}"
        TDA_CLUSTER_FEATURES[f"cluster_{i}"] = members

    TDA_ISOLATED_FEATURES = sorted(isolated)

    print(f"\n  找到 {len(TDA_CLUSTER_FEATURES)} 个族群, {len(TDA_ISOLATED_FEATURES)} 个孤立特征:")
    for name, members in TDA_CLUSTER_FEATURES.items():
        print(f"    {name}: {len(members)} 成员 → {members}")
    print(f"    isolated: {TDA_ISOLATED_FEATURES}")

    # 6. 对每个族群计算 PCA 第一主成分
    print("\n  计算各族群 PCA 第一主成分...")
    for name, members in TDA_CLUSTER_FEATURES.items():
        if len(members) == 1:
            # 单个特征不需要 PCA
            TDA_PCA_WEIGHTS[name] = np.array([1.0])
            continue

        # 提取族群数据
        cluster_data = sample.select(members).to_numpy().astype(np.float64)
        cluster_data = np.nan_to_num(cluster_data, nan=0.0)

        # 标准化
        mean = cluster_data.mean(axis=0, keepdims=True)
        std = cluster_data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        cluster_data = (cluster_data - mean) / std

        # 协方差矩阵 → 第一特征向量
        cov = np.cov(cluster_data, rowvar=False)  # (k, k)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, -1]  # 最大特征值对应的特征向量

        # 归一化为单位向量
        pc1 = pc1 / np.linalg.norm(pc1)
        TDA_PCA_WEIGHTS[name] = pc1

        # 解释方差
        explained_var = eigenvalues[-1] / eigenvalues.sum()
        print(f"    {name}: PC1 解释方差 = {explained_var:.3f}, "
              f"weights = {dict(zip(members, [f'{w:.3f}' for w in pc1]))}")

    # 7. 构建 42 维特征名列表
    TDA_42_FEATURES = []
    # PCA 主成分（每个族群一个）
    for name in sorted(TDA_CLUSTER_FEATURES.keys()):
        TDA_42_FEATURES.append(f"tda_{name}_pc1")
    # 孤立连续特征
    TDA_42_FEATURES.extend(TDA_ISOLATED_FEATURES)
    # 分类特征
    TDA_42_FEATURES.extend(CAT_FEATURES)
    # Lag 特征
    TDA_42_FEATURES.extend(LAG_COLS)

    elapsed = time.time() - t0
    print(f"\n  TDA 42 维特征集: {len(TDA_42_FEATURES)} 维")
    print(f"  耗时: {elapsed:.1f}s")

    return TDA_42_FEATURES


# ============================================================
# 数据准备：树模型（CatBoost / XGBoost）
# ============================================================

def prepare_tree_data(
    sample_rate: int = 6,
    train_end_date: int = TRAIN_END_DATE,
    one_hot: bool = True,
):
    """
    为树模型准备数据，返回 numpy 数组。

    参数:
        sample_rate: 1/N 均匀采样率（N=6 → 约 6.5M 行训练）
        train_end_date: 训练集截止日期
        one_hot: 是否对分类特征做 one-hot 编码

    返回:
        X_train, y_train, w_train: 训练数据
        X_val, y_val, w_val, val_sids: 验证数据
        feature_names: 列名列表
        cat_indices: 分类特征索引（如果不 one-hot）
    """
    print("=" * 60)
    print(f"准备树模型数据 (sample_rate=1/{sample_rate})")
    print("=" * 60)

    t0 = time.time()

    # 加载 + 采样
    train_lazy = _get_train_lazy()
    train_df = train_lazy.collect()
    train_df = train_df.filter(
        pl.int_range(0, pl.len()) % sample_rate == 0
    )
    print(f"  训练集采样: {train_df.height:,} rows")

    val_lazy = _get_val_lazy()
    val_df = val_lazy.collect()
    print(f"  验证集: {val_df.height:,} rows")

    # 连续特征 + lag
    base_features = CONTINUOUS_FEATURES + LAG_COLS

    # 构建 X
    # 连续特征：转换为 float64 numpy
    X_train_parts = []
    X_val_parts = []
    feat_names = list(base_features)

    for c in base_features:
        col_train = train_df[c].to_numpy().astype(np.float32)
        col_val = val_df[c].to_numpy().astype(np.float32)
        col_train = np.nan_to_num(col_train, nan=0.0)
        col_val = np.nan_to_num(col_val, nan=0.0)
        X_train_parts.append(col_train.reshape(-1, 1))
        X_val_parts.append(col_val.reshape(-1, 1))

    cat_indices = []
    if one_hot:
        # One-hot 编码分类特征
        for cat in CAT_FEATURES:
            col_train = train_df[cat].to_numpy()
            col_val = val_df[cat].to_numpy()
            unique_vals = sorted(set(int(v) for v in np.unique(col_train) if not np.isnan(v)))
            for uv in unique_vals:
                oh_train = (col_train == uv).astype(np.float32).reshape(-1, 1)
                oh_val = (col_val == uv).astype(np.float32).reshape(-1, 1)
                X_train_parts.append(oh_train)
                X_val_parts.append(oh_val)
                feat_names.append(f"{cat}_{uv}")
    else:
        # 保持为整数（CatBoost 原生支持）
        for cat in CAT_FEATURES:
            col_train = train_df[cat].to_numpy().astype(np.float32).reshape(-1, 1)
            col_val = val_df[cat].to_numpy().astype(np.float32).reshape(-1, 1)
            col_train = np.nan_to_num(col_train, nan=0.0)
            col_val = np.nan_to_num(col_val, nan=0.0)
            X_train_parts.append(col_train)
            X_val_parts.append(col_val)
            cat_indices.append(len(feat_names))
            feat_names.append(cat)

    X_train = np.hstack(X_train_parts)
    y_train = train_df[TARGET_COL].to_numpy().astype(np.float64)
    w_train = train_df[WEIGHT_COL].to_numpy().astype(np.float64)
    y_train = np.nan_to_num(y_train, nan=0.0)
    w_train = np.nan_to_num(w_train, nan=0.0)

    X_val = np.hstack(X_val_parts)
    y_val = val_df[TARGET_COL].to_numpy().astype(np.float64)
    w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float64)
    val_sids = val_df["symbol_id"].to_numpy()
    y_val = np.nan_to_num(y_val, nan=0.0)
    w_val = np.nan_to_num(w_val, nan=0.0)

    # 释放 polars dataframe 内存
    del train_df, val_df, X_train_parts, X_val_parts
    gc.collect()

    elapsed = time.time() - t0
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"  特征数: {len(feat_names)}")
    print(f"  耗时: {elapsed:.1f}s")

    return X_train, y_train, w_train, X_val, y_val, w_val, val_sids, feat_names, cat_indices


# ============================================================
# 数据准备：MLP（PyTorch DataLoader）
# ============================================================

class MLPDataset:
    """
    MLP 训练数据集（不继承 torch Dataset，避免序列化开销）。
    数据存储在 float32 numpy 数组中，__getitem__ 返回 torch tensor。
    支持两种模式：
    - in_memory: 全部加载到 RAM
    - memmap: 磁盘映射（适合 RAM 不足的情况）
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = X
        self.y = y
        self.w = w
        self._len = len(y)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # 返回 CPU tensor，DataLoader 会负责 batch + transfer
        import torch
        return (
            torch.from_numpy(self.X[idx].astype(np.float32)),
            torch.tensor(float(self.y[idx]), dtype=torch.float32),
            torch.tensor(float(self.w[idx]), dtype=torch.float32),
        )


def _load_and_normalize(lazy_df, feature_cols, stats):
    """
    加载 LazyFrame → numpy float32，应用标准化。
    分块加载以避免内存峰值。
    """
    chunk_size = 2_000_000
    n_total = lazy_df.select(pl.len()).collect().item()

    # 估算内存
    est_mem_gb = n_total * len(feature_cols) * 4 / 1e9
    print(f"  估计需要 {est_mem_gb:.1f} GB (float32)")

    # 分配数组
    X = np.empty((n_total, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_total, dtype=np.float32)
    w = np.empty(n_total, dtype=np.float32)

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        offset = i * chunk_size
        chunk = lazy_df.slice(offset, chunk_size).collect()
        n = chunk.height
        if n == 0:
            break

        # 标准化连续特征
        for j, c in enumerate(feature_cols):
            vals = chunk[c].to_numpy().astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            if c in stats:
                m, s = stats[c]["mean"], stats[c]["std"]
                vals = (vals - m) / s
            X[offset:offset + n, j] = vals

        y[offset:offset + n] = np.nan_to_num(
            chunk[TARGET_COL].to_numpy().astype(np.float32), nan=0.0
        )
        w[offset:offset + n] = np.nan_to_num(
            chunk[WEIGHT_COL].to_numpy().astype(np.float32), nan=0.0
        )

        if i % 10 == 0:
            print(f"  加载进度: {offset + n:,}/{n_total:,}")

    return X, y, w


def _build_tda_features(X_full, feature_cols, stats):
    """
    从全量 96 维特征构建 TDA 42 维特征。
    必须先在 compute_tda_clusters() 中填充全局变量。
    """
    n_samples = X_full.shape[0]
    n_tda = len(TDA_42_FEATURES)
    X_tda = np.empty((n_samples, n_tda), dtype=np.float32)

    # 构建 feature_col → column index 映射
    col_map = {c: i for i, c in enumerate(feature_cols)}

    write_idx = 0

    # 1. PCA 主成分
    for name in sorted(TDA_CLUSTER_FEATURES.keys()):
        members = TDA_CLUSTER_FEATURES[name]
        weights = TDA_PCA_WEIGHTS[name]

        # 提取族群数据并标准化
        cluster_data = np.column_stack([
            X_full[:, col_map[m]] for m in members
        ])

        # 用全局统计量标准化
        for j, m in enumerate(members):
            if m in stats:
                mu, sg = stats[m]["mean"], stats[m]["std"]
                cluster_data[:, j] = (cluster_data[:, j] - mu) / sg

        # 投影到 PC1
        pc1 = cluster_data @ weights
        X_tda[:, write_idx] = pc1.astype(np.float32)
        write_idx += 1

    # 2. 孤立特征（已标准化）
    for feat in TDA_ISOLATED_FEATURES:
        X_tda[:, write_idx] = X_full[:, col_map[feat]]
        write_idx += 1

    # 3. 分类特征
    for feat in CAT_FEATURES:
        X_tda[:, write_idx] = X_full[:, col_map[feat]]
        write_idx += 1

    # 4. Lag 特征
    for feat in LAG_COLS:
        X_tda[:, write_idx] = X_full[:, col_map[feat]]
        write_idx += 1

    return X_tda


def _add_regime_features(
    X: np.ndarray,
    symbol_ids: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    在标准化后的特征矩阵上添加 Δfeatures 和 regime_score。

    Δfeature: 每个 symbol 内相邻时间步的特征变化（一阶差分）
    regime_score: 5 个关键特征偏离均值的程度之和
        → score 高 = 当前数据离训练分布远 = regime shift 信号
    """
    key_indices = [feature_names.index(f) for f in REGIME_KEY_FEATURES if f in feature_names]
    n = X.shape[0]
    n_delta = len(key_indices)

    # ---- Δfeatures: per-symbol 一阶差分 ----
    delta = np.zeros((n, n_delta), dtype=np.float32)
    unique_sids = np.unique(symbol_ids)
    for i, feat_idx in enumerate(key_indices):
        col = X[:, feat_idx]
        for sid in unique_sids:
            mask = symbol_ids == sid
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue
            vals = col[idx]
            diff = np.zeros(len(vals), dtype=np.float32)
            diff[1:] = vals[1:] - vals[:-1]
            delta[idx, i] = diff.astype(np.float32)

    # ---- regime_score: sum of |z-score| of key features ----
    regime = np.abs(X[:, key_indices].astype(np.float64)).sum(axis=1, keepdims=True).astype(np.float32)

    X_aug = np.column_stack([X, delta, regime])
    new_names = REGIME_DELTA_NAMES + REGIME_SCORE_NAME
    return X_aug, new_names


def prepare_mlp_data(feature_set: str = "full", sample_rate: int = 1,
                     use_regime: bool = True, use_ricci: bool = False):
    """
    为 MLP 准备数据。

    参数:
        feature_set: "full" (96维) 或 "tda" (42维)
        sample_rate: 1/N 采样率（1=全量，3=1/3，降低 RAM 占用）
        use_regime: 是否添加 torsion 驱动的 regime 特征（Δ + regime_score）
        use_ricci: 是否用 Ricci 曲率调节样本权重

    返回:
        (X_train, y_train, w_train), (X_val, y_val, w_val), feature_names
    """
    print("=" * 60)
    print(f"准备 MLP 数据 (feature_set={feature_set})")
    print("=" * 60)

    t0 = time.time()

    feature_cols = FULL_FEATURES_96
    if feature_set == "tda":
        if not TDA_42_FEATURES:
            compute_tda_clusters()
        feature_cols = FULL_FEATURES_96  # 先加载全量，再降维

    # 流式计算统计量
    print("  计算特征统计量...")
    train_lazy = _get_train_lazy()
    stats = compute_stats(train_lazy, FULL_FEATURES_96)

    # 采样训练集（减少 RAM，统计量仍用全量数据计算）
    if sample_rate > 1:
        train_lazy = train_lazy.filter(
            pl.int_range(0, pl.len()) % sample_rate == 0
        )
        print(f"  训练集采样率: 1/{sample_rate}")

    # 加载训练集
    print("  加载训练集...")
    X_train_full, y_train, w_train = _load_and_normalize(train_lazy, FULL_FEATURES_96, stats)

    # 加载验证集
    print("  加载验证集...")
    val_lazy = _get_val_lazy()
    X_val_full, y_val, w_val = _load_and_normalize(val_lazy, FULL_FEATURES_96, stats)

    # ---- Regime 特征（torsion 驱动）----
    train_sids = train_lazy.select("symbol_id").collect()["symbol_id"].to_numpy()
    val_sids = val_lazy.select("symbol_id").collect()["symbol_id"].to_numpy()

    if use_regime:
        print("  计算 regime 感知特征...")
        X_train_regime, regime_names = _add_regime_features(X_train_full, train_sids, FULL_FEATURES_96)
        X_val_regime, _ = _add_regime_features(X_val_full, val_sids, FULL_FEATURES_96)
        n_regime = len(regime_names)
        print(f"  新增 {n_regime} 维: {regime_names}")
    else:
        regime_names = []

    if feature_set == "tda":
        print("  构建 TDA 42 维特征...")
        X_train_base = _build_tda_features(X_train_full, FULL_FEATURES_96, stats)
        X_val_base = _build_tda_features(X_val_full, FULL_FEATURES_96, stats)
        base_names = TDA_42_FEATURES
    else:
        X_train_base = X_train_full
        X_val_base = X_val_full
        base_names = FULL_FEATURES_96

    # 拼接 regime 特征（如果需要）
    if use_regime:
        X_train = np.column_stack([X_train_base, X_train_regime[:, -n_regime:]])
        X_val = np.column_stack([X_val_base, X_val_regime[:, -n_regime:]])
        feat_names = base_names + regime_names
    else:
        X_train = X_train_base
        X_val = X_val_base
        feat_names = base_names

    # 释放中间数组
    del X_train_full, X_val_full, X_train_base, X_val_base
    if use_regime:
        del X_train_regime, X_val_regime
    gc.collect()

    # ---- Ricci 曲率权重 ----
    if use_ricci:
        print("  计算 Ricci 曲率样本权重...")
        ricci_w_train = _compute_ricci_weights(y_train, train_sids)
        ricci_w_val = _compute_ricci_weights(y_val, val_sids)
        w_train = (w_train * ricci_w_train).astype(np.float32)
        w_val = (w_val * ricci_w_val).astype(np.float32)
        print(f"  训练权重: mean={ricci_w_train.mean():.3f}, "
              f"range=[{ricci_w_train.min():.2f}, {ricci_w_train.max():.2f}]")
        print(f"  验证权重: mean={ricci_w_val.mean():.3f}, "
              f"range=[{ricci_w_val.min():.2f}, {ricci_w_val.max():.2f}]")

    elapsed = time.time() - t0
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"  特征数: {len(feat_names)}")
    print(f"  耗时: {elapsed:.1f}s")

    return (X_train, y_train, w_train), (X_val, y_val, w_val), feat_names


# ============================================================
# 工具：创建 models 目录 & 内存信息
# ============================================================

def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def print_memory_info():
    """打印系统和 GPU 内存信息"""
    print("\n--- 内存信息 ---")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  系统 RAM: {mem.total / 1e9:.1f} GB total, "
              f"{mem.available / 1e9:.1f} GB available")
    except ImportError:
        print("  (psutil 未安装，无法获取系统内存)")

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory / 1024**2
                print(f"  GPU {i}: {props.name}, {total_mb:.0f} MB VRAM")
    except ImportError:
        print("  (PyTorch 未安装)")
    print("---\n")
