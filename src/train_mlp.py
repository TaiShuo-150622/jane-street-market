"""
MLP PyTorch GPU 训练
====================
- 支持两个特征集：full (96维) 和 tda (42维)
- 架构监控 + 自动降级
- AdamW + ReduceLROnPlateau
- 最优模型 checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import gc
import json

from .data_utils import (
    prepare_mlp_data, MODELS_DIR, TARGET_COL, WEIGHT_COL,
    FULL_FEATURES_96, TDA_42_FEATURES, compute_tda_clusters,
)
from .metrics import weighted_r2, weighted_r2_per_group


# ============================================================
# Dataset
# ============================================================

class JaneStreetDataset(Dataset):
    """CPU 端存储，__getitem__ 返回单行 tensor"""

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = X  # float32, (n, d)
        self.y = y  # float32, (n,)
        self.w = w  # float32, (n,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.w[idx], dtype=torch.float32),
        )


# ============================================================
# Model
# ============================================================

class MLP(nn.Module):
    """
    ResNet-style MLP with BatchNorm + SiLU + Dropout.
    Output: Tanh() * 5 (matches target clip range)
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropouts: list[float]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return 5.0 * self.net(x).squeeze(-1)


# ============================================================
# Training loop
# ============================================================

def train(
    feature_set: str = "full",
    hidden_dims: list[int] | None = None,
    dropouts: list[float] | None = None,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 25,
    num_workers: int = 0,
    save_prefix: str = "mlp",
    resume: bool = False,
    cache_data: bool = True,
    sample_rate: int = 1,
    use_regime: bool = True,
    use_ricci: bool = False,
):
    """
    训练 MLP 模型（GPU）。

    参数:
        feature_set: "full" (96维) 或 "tda" (42维)
        hidden_dims: 隐藏层维度，默认 [512, 512, 256]
        dropouts: Dropout 率，默认 [0.1, 0.1]
        batch_size: 批量大小
        lr: 初始学习率
        weight_decay: AdamW weight decay
        max_epochs: 最大 epoch
        patience: 早停耐心值
        num_workers: DataLoader 工作线程（Windows 建议 0）
        resume: 是否从 checkpoint 恢复训练
        cache_data: 是否缓存标准化后的数据到 disk（加速 resume）
        sample_rate: 1/N 训练集采样率（1=全量，3=1/3，降低 RAM）
        use_regime: 是否添加 torsion 驱动的 regime 特征
        use_ricci: 是否用 Ricci 曲率调节样本权重
    """
    if hidden_dims is None:
        hidden_dims = [512, 512, 256]
    if dropouts is None:
        dropouts = [0.1, 0.1]

    model_path = MODELS_DIR / f"{save_prefix}_{feature_set}.pth"
    result_path = MODELS_DIR / f"{save_prefix}_{feature_set}_results.json"
    data_cache_dir = MODELS_DIR / "mlp_cache"
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = data_cache_dir / f"{feature_set}"

    print("=" * 60)
    print(f"MLP 训练: feature_set={feature_set}")
    print(f"  架构: {hidden_dims}, Dropout: {dropouts}")
    print(f"  Batch: {batch_size}, LR: {lr}, WD: {weight_decay}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory / 1024**2:.0f} MB")

    t0 = time.time()

    # ---- 1. 准备数据 ----
    print("\n[1/4] 加载数据...")

    # 尝试读缓存
    cache_X_train = data_cache_dir / f"{feature_set}_X_train.npy"
    cache_y_train = data_cache_dir / f"{feature_set}_y_train.npy"

    if cache_data and cache_X_train.exists() and cache_y_train.exists():
        print("  从缓存加载训练数据...")
        X_train = np.load(cache_X_train, mmap_mode="r")
        y_train = np.load(cache_y_train, mmap_mode="r")
        w_train = np.load(data_cache_dir / f"{feature_set}_w_train.npy", mmap_mode="r")

        # 验证集总是加载（小得多）
        val_lazy_cache = data_cache_dir / f"{feature_set}_X_val.npy"
        if val_lazy_cache.exists():
            X_val = np.load(val_lazy_cache)
            y_val = np.load(data_cache_dir / f"{feature_set}_y_val.npy")
            w_val = np.load(data_cache_dir / f"{feature_set}_w_val.npy")
            feat_names = FULL_FEATURES_96 if feature_set == "full" else TDA_42_FEATURES
        else:
            _, (X_val, y_val, w_val), feat_names = prepare_mlp_data(feature_set, sample_rate, use_regime, use_ricci)
    else:
        (X_train, y_train, w_train), (X_val, y_val, w_val), feat_names = prepare_mlp_data(feature_set, sample_rate, use_regime, use_ricci)

        # 写缓存（加速下次 load / resume）
        if cache_data:
            print("  写数据缓存...")
            np.save(cache_X_train, X_train)
            np.save(cache_y_train, y_train)
            np.save(data_cache_dir / f"{feature_set}_w_train.npy", w_train)
            np.save(data_cache_dir / f"{feature_set}_X_val.npy", X_val)
            np.save(data_cache_dir / f"{feature_set}_y_val.npy", y_val)
            np.save(data_cache_dir / f"{feature_set}_w_val.npy", w_val)

    input_dim = X_train.shape[1]
    print(f"  输入维度: {input_dim}")

    train_dataset = JaneStreetDataset(X_train, y_train, w_train)
    val_dataset = JaneStreetDataset(X_val, y_val, w_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- 2. 模型 & 优化器 ----
    print("\n[2/4] 创建模型...")

    start_epoch = 0
    if resume and model_path.exists():
        print("  从 checkpoint 恢复训练...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        saved_hidden_dims = checkpoint.get("hidden_dims", hidden_dims)
        saved_input_dim = checkpoint.get("input_dim", input_dim)
        model = MLP(saved_input_dim, saved_hidden_dims, dropouts).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_val_r2 = checkpoint.get("val_r2", -np.inf)
        print(f"  恢复: epoch={checkpoint['epoch']+1}, val_R²={saved_val_r2:.6f}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                print("  ⚠ optimizer state 不兼容，使用新优化器")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8,
            min_lr=1e-6,
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "val_r2": []})
        best_val_r2 = saved_val_r2
        best_epoch = start_epoch
        epochs_no_improve = 0
    else:
        model = MLP(input_dim, hidden_dims, dropouts).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8,
            min_lr=1e-6,
        )
        best_val_r2 = -np.inf
        best_epoch = 0
        history = {"train_loss": [], "val_loss": [], "val_r2": []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # ---- 3. 训练循环（含监控） ----
    print(f"\n[3/4] 训练 (从 epoch {start_epoch + 1} 开始)...")

    epochs_no_improve = 0

    # 监控状态
    monitor_lr_reduced = False
    monitor_batch_reduced = False
    monitor_bad_r2_count = 0
    monitor_max_bad_r2 = 5  # 连续 5 epoch val_R² < 0 则降级

    for epoch in range(start_epoch, max_epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = (F.mse_loss(pred, batch_y, reduction="none") * batch_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        all_preds, all_y, all_w = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_w in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                pred = model(batch_x)
                loss = (F.mse_loss(pred, batch_y.to(device), reduction="none")
                        * batch_w.to(device)).mean()
                val_loss += loss.item() * batch_x.size(0)
                all_preds.append(pred.cpu().numpy())
                all_y.append(batch_y.numpy())
                all_w.append(batch_w.numpy())

        val_loss /= len(val_dataset)
        y_pred_val = np.concatenate(all_preds)
        val_r2 = weighted_r2(np.concatenate(all_y), y_pred_val, np.concatenate(all_w))

        scheduler.step(val_loss)

        # 日志
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(float(val_r2))

        lr_now = optimizer.param_groups[0]["lr"]
        is_best = "← BEST" if val_r2 > best_val_r2 else ""
        print(f"  Epoch {epoch+1:3d}/{max_epochs} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"val_R²={val_r2:+.6f} | lr={lr_now:.2e} {is_best}")

        # ---- 监控 ----
        # 1. Early epochs: loss 不降 → 减半 LR
        if epoch == 10 and not monitor_lr_reduced:
            early_losses = history["val_loss"][:10]
            if len(early_losses) >= 5 and early_losses[-1] >= early_losses[0] * 0.95:
                print("  ⚠ 监控: val_loss 前10epoch未显著下降 → 学习率减半")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5
                monitor_lr_reduced = True

        # 2. VRAM 监控
        if epoch == 3 and not monitor_batch_reduced and device.type == "cuda":
            vram_used = torch.cuda.memory_allocated() / 1024**3
            if vram_used > 10.0:
                print(f"  ⚠ 监控: VRAM 使用 {vram_used:.1f} GB > 10GB → batch_size 减半")
                batch_size = max(1024, batch_size // 2)
                # 重建 DataLoader
                del train_loader, val_loader
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True,
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size * 2, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
                monitor_batch_reduced = True

        # 3. 负 R² 监控 → 降级架构
        if val_r2 < 0:
            monitor_bad_r2_count += 1
            if monitor_bad_r2_count >= monitor_max_bad_r2 and hidden_dims != [512, 256, 128]:
                print("  ⚠ 监控: val_R² 连续5 epoch < 0 → 降级架构 [512,256,128]")
                # 重置并重启
                del model
                torch.cuda.empty_cache()
                hidden_dims = [512, 256, 128]
                model = MLP(input_dim, hidden_dims, dropouts).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=8,
                    min_lr=1e-6,
                )
                best_val_r2 = -np.inf
                epochs_no_improve = 0
                monitor_bad_r2_count = 0
                continue
        else:
            monitor_bad_r2_count = 0

        # ---- Checkpoint ----
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_r2": val_r2,
                "hidden_dims": hidden_dims,
                "input_dim": input_dim,
                "history": history,
            }, model_path)
        else:
            epochs_no_improve += 1

        # ---- Early stopping ----
        if epochs_no_improve >= patience:
            print(f"  Early stopping: {patience} epoch 无提升")
            break

    # ---- 4. 评估 ----
    print(f"\n[4/4] 评估 (最佳 epoch={best_epoch})...")

    # 加载最佳模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 全量验证集预测
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, batch_y, batch_w in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            pred = model(batch_x).cpu().numpy()
            all_preds.append(pred)
    y_pred_val = np.concatenate(all_preds)

    val_r2 = weighted_r2(y_val, y_pred_val, w_val)
    naive_r2 = weighted_r2(y_val, np.zeros_like(y_val), w_val)

    print(f"  验证 R² = {val_r2:.6f}")
    print(f"  全 0 基线 = {naive_r2:.6f}")
    print(f"  相对提升 = {val_r2 - naive_r2:+.6f}")

    # 保存结果
    results = {
        "model": f"MLP_{feature_set}",
        "feature_set": feature_set,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "batch_size": batch_size,
        "n_params": n_params,
        "val_r2": float(val_r2),
        "naive_r2": float(naive_r2),
        "delta": float(val_r2 - naive_r2),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "train_time_s": time.time() - t0,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n  ✓ MLP ({feature_set}) 完成 | R²={val_r2:.6f} | 耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # ---- R² 收敛曲线 ----
    if history["val_r2"]:
        _print_r2_progression(history, best_epoch)

    # 保存验证集预测（用于集成）
    preds_path = MODELS_DIR / f"{save_prefix}_{feature_set}_val_preds.npy"
    np.save(preds_path, y_pred_val.astype(np.float32))

    # 清理
    del model, train_dataset, val_dataset, X_train, y_train, w_train, X_val, y_val, w_val
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_r2


def train_both():
    """依次训练 MLP v1 (96维) 和 v2 (42维)"""
    results = {}

    # v1: 全量 96 维
    print("\n" + "=" * 70)
    print("MLP v1: 全量 96 维特征")
    print("=" * 70)
    r2_v1 = train(feature_set="full", save_prefix="mlp")
    results["mlp_full"] = r2_v1
    _clear_gpu()

    # v2: TDA 42 维
    print("\n" + "=" * 70)
    print("MLP v2: TDA 42 维特征")
    print("=" * 70)
    compute_tda_clusters()  # 确保 TDA 聚类已完成
    r2_v2 = train(feature_set="tda", save_prefix="mlp")
    results["mlp_tda"] = r2_v2
    _clear_gpu()

    print("\n" + "=" * 70)
    print("MLP 训练汇总")
    print(f"  v1 (96dim): R² = {results['mlp_full']:.6f}")
    print(f"  v2 (42dim): R² = {results['mlp_tda']:.6f}")
    print("=" * 70)

    return results


def _print_r2_progression(history: dict, best_epoch: int):
    """打印 R² 收敛曲线，方便判断训练是否到位"""
    val_r2s = history["val_r2"]
    n = len(val_r2s)
    if n == 0:
        return

    # 找关键节点
    best_idx = val_r2s.index(max(val_r2s))
    first_positive = next((i for i, r in enumerate(val_r2s) if r > 0), None)
    last_10 = val_r2s[-min(10, n):]

    print(f"\n  R² 收敛曲线 ({n} epochs):")
    # 打印每 5 epoch + 关键点
    printed = set()
    for i in range(n):
        if i == 0 or i == n - 1 or i == best_idx or (i + 1) % 5 == 0:
            printed.add(i)
    printed = sorted(printed)

    for i in printed:
        marker = ""
        if i == 0:
            marker = " (start)"
        if i == best_idx:
            marker = " ← BEST"
        if i == n - 1:
            marker = " (final)"
        bar = "+" * max(0, int(val_r2s[i] * 500)) if val_r2s[i] > 0 else "-" * min(5, int(abs(val_r2s[i]) * 10))
        print(f"    epoch {i+1:3d}: R²={val_r2s[i]:+.6f}  {bar}{marker}")

    # 收敛判断
    if len(last_10) >= 5:
        range_last_10 = max(last_10) - min(last_10)
        if range_last_10 < 0.0005 and val_r2s[best_idx] > 0:
            print(f"\n  ✅ 最后 10 epoch R² 波动 {range_last_10:.6f} < 0.0005，已收敛")
        elif range_last_10 < 0.001:
            print(f"\n  ⚠️ 最后 10 epoch R² 波动 {range_last_10:.6f}，可能还在缓慢提升")
        else:
            print(f"\n  ❌ 最后 10 epoch R² 波动 {range_last_10:.6f} > 0.001，未收敛，考虑增加 patience")


def _clear_gpu():
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


if __name__ == "__main__":
    train_both()
