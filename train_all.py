#!/usr/bin/env python3
"""
Jane Street Market Prediction — 一键训练脚本
=============================================
顺序训练 CatBoost → MLP v1 → MLP v2 → XGBoost → 集成。
每步独立 try/catch，前面成功的模型不受影响。
每个模型训练完自动保存 checkpoint + 验证集预测。

用法:
    python train_all.py

环境要求:
    - CUDA GPU (已测试 2080 Ti 11GB)
    - Python 3.10+
    - pip install -r requirements_gpu.txt
"""

import sys
import time
import json
import gc
import traceback
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from src.data_utils import (
    MODELS_DIR, TRAIN_PATH, TARGET_COL, WEIGHT_COL, TRAIN_END_DATE,
    ensure_models_dir, print_memory_info,
)
from src.metrics import weighted_r2

# ============================================================
# 配置
# ============================================================

# 是否运行 XGBoost（树模型时间充裕则运行）
RUN_XGBOOST = True

# 采样率（树模型用 1/N 数据，控制内存和时间）
TREE_SAMPLE_RATE = 10  # 1/10 ≈ 3.6M rows（内存紧张可继续调大）

# MLP 训练集采样率（1=全量 ≈36M rows ≈14GB；3=1/3 ≈12M rows ≈5GB）
MLP_SAMPLE_RATE = 1  # 内存紧张建议改为 3

# 集成模式: "grid_search" | "stacking" | "simple_average"
ENSEMBLE_MODE = "grid_search"


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def is_model_done(model_file: str) -> bool:
    """检查模型文件是否已存在（支持断点续跑）"""
    path = MODELS_DIR / model_file
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        print(f"  ⏭ 模型已存在: {model_file} ({size_mb:.1f} MB) → 跳过训练")
        return True
    return False


def _load_model_r2(results_json: str) -> float | None:
    """从 results.json 中读取 val_r2"""
    import json as _json
    path = MODELS_DIR / results_json
    if path.exists():
        with open(path) as f:
            data = _json.load(f)
        return data.get("val_r2")
    return None


def save_val_targets():
    """从验证集中提取并保存 y, w, sids（所有模型验证集行一致）"""
    y_path = MODELS_DIR / "val_y.npy"
    w_path = MODELS_DIR / "val_w.npy"
    sids_path = MODELS_DIR / "val_sids.npy"

    if y_path.exists() and w_path.exists():
        print("  验证集 target 已缓存，跳过")
        return

    print("  提取验证集 target...")
    val_lazy = pl.scan_parquet(TRAIN_PATH).filter(
        pl.col("date_id") > TRAIN_END_DATE
    )
    val_df = val_lazy.collect()
    y_val = val_df[TARGET_COL].to_numpy().astype(np.float64)
    w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float64)
    val_sids = val_df["symbol_id"].to_numpy()
    y_val = np.nan_to_num(y_val, nan=0.0)
    w_val = np.nan_to_num(w_val, nan=0.0)

    np.save(y_path, y_val)
    np.save(w_path, w_val)
    np.save(sids_path, val_sids)
    print(f"  已保存: y={y_val.shape}, w={w_val.shape}, sids={val_sids.shape}")


def step_catboost():
    """Step 1: CatBoost"""
    section("Step 1/5: CatBoost GPU 训练")
    if is_model_done("catboost_model.cbm"):
        r2 = _load_model_r2("catboost_results.json")
        return {"catboost": {"r2": r2, "status": "cached"}}
    try:
        from src.train_catboost import train
        r2 = train(sample_rate=TREE_SAMPLE_RATE)
        print(f"  ✅ CatBoost 完成: R²={r2:.6f}")
        return {"catboost": {"r2": float(r2) if r2 else None, "status": "ok"}}
    except Exception as e:
        print(f"  ❌ CatBoost 失败: {e}")
        traceback.print_exc()
        return {"catboost": {"status": "failed", "error": str(e)}}


def step_mlp_v1():
    """Step 2: MLP v1 (全量 88 维)"""
    section("Step 2/5: MLP v1 (全量 88 维特征)")
    if is_model_done("mlp_full_results.json"):   # results.json 只有训练完整结束才写
        r2 = _load_model_r2("mlp_full_results.json")
        return {"mlp_full": {"r2": r2, "status": "cached"}}
    try:
        from src.train_mlp import train
        r2 = train(feature_set="full", save_prefix="mlp", resume=True, sample_rate=MLP_SAMPLE_RATE)
        print(f"  ✅ MLP v1 完成: R²={r2:.6f}")
        return {"mlp_full": {"r2": float(r2) if r2 else None, "status": "ok"}}
    except Exception as e:
        print(f"  ❌ MLP v1 失败: {e}")
        traceback.print_exc()
        _emergency_gpu_cleanup()
        return {"mlp_full": {"status": "failed", "error": str(e)}}


def step_mlp_v2():
    """Step 3: MLP v2 (TDA 42 维)"""
    section("Step 3/5: MLP v2 (TDA 42 维特征)")
    if is_model_done("mlp_tda_results.json"):   # results.json only written after full completion
        r2 = _load_model_r2("mlp_tda_results.json")
        return {"mlp_tda": {"r2": r2, "status": "cached"}}
    try:
        from src.data_utils import compute_tda_clusters
        compute_tda_clusters()  # 确保 TDA 聚类完成
        from src.train_mlp import train
        r2 = train(feature_set="tda", save_prefix="mlp", resume=True, sample_rate=MLP_SAMPLE_RATE)
        print(f"  ✅ MLP v2 完成: R²={r2:.6f}")
        return {"mlp_tda": {"r2": float(r2) if r2 else None, "status": "ok"}}
    except Exception as e:
        print(f"  ❌ MLP v2 失败: {e}")
        traceback.print_exc()
        _emergency_gpu_cleanup()
        return {"mlp_tda": {"status": "failed", "error": str(e)}}


def step_xgboost():
    """Step 4: XGBoost"""
    section("Step 4/5: XGBoost GPU 训练")
    if is_model_done("xgboost_model.json"):
        r2 = _load_model_r2("xgboost_results.json")
        return {"xgboost": {"r2": r2, "status": "cached"}}
    try:
        from src.train_xgboost import train
        r2 = train(sample_rate=TREE_SAMPLE_RATE)
        print(f"  ✅ XGBoost 完成: R²={r2:.6f}")
        return {"xgboost": {"r2": float(r2) if r2 else None, "status": "ok"}}
    except Exception as e:
        print(f"  ❌ XGBoost 失败: {e}")
        traceback.print_exc()
        _emergency_gpu_cleanup()
        return {"xgboost": {"status": "failed", "error": str(e)}}


def step_tabm():
    """Step 5: TabM (官方包)"""
    section("Step 5/6: TabM (官方 tabm 包)")
    if is_model_done("tabm_results.json"):
        r2 = _load_model_r2("tabm_results.json")
        return {"tabm": {"r2": r2, "status": "cached"}}
    try:
        from src.train_tabm import train
        r2 = train()
        print(f"  ✅ TabM 完成: R²={r2:.6f}")
        return {"tabm": {"r2": float(r2) if r2 else None, "status": "ok"}}
    except Exception as e:
        print(f"  ❌ TabM 失败: {e}")
        traceback.print_exc()
        _emergency_gpu_cleanup()
        return {"tabm": {"status": "failed", "error": str(e)}}


def step_ensemble():
    """Step 6: 集成权重搜索"""
    section("Step 6/6: 集成权重搜索")

    # 收集可用的预测
    pred_files = {
        "catboost": MODELS_DIR / "catboost_val_preds.npy",
        "mlp_full": MODELS_DIR / "mlp_full_val_preds.npy",
        "mlp_tda": MODELS_DIR / "mlp_tda_val_preds.npy",
        "xgboost": MODELS_DIR / "xgboost_val_preds.npy",
        "tabm": MODELS_DIR / "tabm_val_preds.npy",
    }

    y_path = MODELS_DIR / "val_y.npy"
    w_path = MODELS_DIR / "val_w.npy"
    sids_path = MODELS_DIR / "val_sids.npy"

    if not y_path.exists():
        print("  ❌ 验证集 target 未找到！请确保前面的步骤至少有一个成功。")
        return {"ensemble": {"status": "failed", "error": "no val targets"}}

    y_val = np.load(y_path)
    w_val = np.load(w_path)
    val_sids = np.load(sids_path)

    predictions = {}
    for name, path in pred_files.items():
        if path.exists():
            preds = np.load(path).astype(np.float64).ravel()
            # 验证长度一致
            if len(preds) != len(y_val):
                print(f"  ⚠ {name}: 预测长度 {len(preds)} != val 长度 {len(y_val)}，跳过")
                continue
            predictions[name] = preds
            r2 = weighted_r2(y_val, preds, w_val)
            print(f"  {name}: R²={r2:.6f} (已加载)")
        else:
            print(f"  {name}: 预测文件不存在，跳过")

    if len(predictions) < 2:
        print(f"  ❌ 可用模型不足 ({len(predictions)} < 2)，无法集成")
        return {"ensemble": {"status": "failed", "error": "too few models"}}

    # 集成
    from src.ensemble import load_and_ensemble
    result = load_and_ensemble(
        predictions, y_val, w_val, val_sids,
        mode=ENSEMBLE_MODE, grid_step=0.05,
    )
    print(f"  ✅ 集成完成")
    return {"ensemble": {"status": "ok", **result}}


def _emergency_gpu_cleanup():
    """紧急 GPU 清理"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    gc.collect()


# ============================================================
# Main
# ============================================================

def main():
    global_start = time.time()

    print("=" * 70)
    print("  Jane Street Market Prediction — 一键训练")
    print("=" * 70)

    # 系统信息
    print_memory_info()
    ensure_models_dir()

    # 多 GPU 检测
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"  检测到 {n_gpus} 张 GPU，将启用多 GPU 训练")
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name}, {props.total_memory/1024**3:.1f} GB")
    except ImportError:
        pass

    print(f"  模型输出目录: {MODELS_DIR}")
    print(f"  数据文件: {TRAIN_PATH}")
    print(f"  集成模式: {ENSEMBLE_MODE}")
    print(f"  树模型采样率: 1/{TREE_SAMPLE_RATE}")
    print(f"  运行 XGBoost: {RUN_XGBOOST}")

    if not TRAIN_PATH.exists():
        print(f"\n  ❌ 错误: 数据文件不存在: {TRAIN_PATH}")
        print(f"  请先运行 src/preprocess.py 生成预处理数据。")
        sys.exit(1)

    # 保存验证集 target（所有模型共用）
    section("准备: 提取验证集 target")
    save_val_targets()

    # 汇总结果
    all_results = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(TRAIN_PATH),
        "models_dir": str(MODELS_DIR),
        "ensemble_mode": ENSEMBLE_MODE,
        "tree_sample_rate": TREE_SAMPLE_RATE,
    }

    # ---- Step 1: CatBoost ----
    all_results.update(step_catboost())

    # ---- Step 2: MLP v1 ----
    all_results.update(step_mlp_v1())

    # ---- Step 3: MLP v2 ----
    all_results.update(step_mlp_v2())

    # ---- Step 4: XGBoost (optional) ----
    if RUN_XGBOOST:
        all_results.update(step_xgboost())
    else:
        print("\n  ⏭ 跳过 XGBoost (RUN_XGBOOST=False)")

    # ---- Step 5: TabM ----
    all_results.update(step_tabm())

    # ---- Step 6: Ensemble ----
    all_results.update(step_ensemble())

    # ---- 终报 ----
    total_time = time.time() - global_start
    all_results["total_time_s"] = total_time

    section("训练完成")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} h)")
    print(f"\n  各模型结果:")
    for model_name in ["catboost", "mlp_full", "mlp_tda", "xgboost", "tabm"]:
        if model_name in all_results:
            r = all_results[model_name]
            if r.get("status") == "ok" and r.get("r2") is not None:
                print(f"    ✅ {model_name:12s}: R²={r['r2']:+.6f}")
            else:
                print(f"    ❌ {model_name:12s}: {r.get('error', 'failed')}")

    # 保存最终报告
    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  完整报告: {report_path}")

    # 列出输出文件
    print(f"\n  输出文件:")
    for f in sorted(MODELS_DIR.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name:40s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
