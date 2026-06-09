#!/usr/bin/env python3
"""
MLP 消融实验 — 衡量 torsion 和 Ricci 各自对 R² 的贡献
======================================================
4 个配置依次训练，最后对比 R²。

用法:
    python ablation_mlp.py
"""
import sys, time, json, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.train_mlp import train
from src.metrics import weighted_r2

CONFIGS = [
    {"label": "A: 基线 (88维, 无任何增强)",       "use_regime": False, "use_ricci": False, "prefix": "ablation_a_baseline"},
    {"label": "B: 仅 torsion (94维, +Δ+regime)", "use_regime": True,  "use_ricci": False, "prefix": "ablation_b_regime"},
    {"label": "C: 仅 Ricci (88维, 权重调节)",     "use_regime": False, "use_ricci": True,  "prefix": "ablation_c_ricci"},
    {"label": "D: 两者都加 (94维, +Δ+regime+Ricci)", "use_regime": True, "use_ricci": True, "prefix": "ablation_d_both"},
]

results = {}
t_start = time.time()

for cfg in CONFIGS:
    print("\n" + "=" * 70)
    print(f"  {cfg['label']}")
    print("=" * 70)

    r2 = train(
        feature_set="full",
        sample_rate=1,
        use_regime=cfg["use_regime"],
        use_ricci=cfg["use_ricci"],
        save_prefix=cfg["prefix"],
        resume=False,
        cache_data=False,
    )
    results[cfg["prefix"]] = {
        "label": cfg["label"],
        "use_regime": cfg["use_regime"],
        "use_ricci": cfg["use_ricci"],
        "r2": float(r2) if r2 is not None else None,
    }
    gc.collect()

# ---- 对比报告 ----
print("\n" + "=" * 70)
print("  消融实验报告")
print("=" * 70)

print(f"\n{'配置':<40} {'R²':>10}  {'vs 基线':>10}")
print("-" * 62)

baseline_r2 = results["ablation_a_baseline"]["r2"]
for cfg in CONFIGS:
    r = results[cfg["prefix"]]
    r2_str = f"{r['r2']:+.6f}" if r["r2"] is not None else "FAILED"
    if r["r2"] is not None and baseline_r2 is not None and cfg["prefix"] != "ablation_a_baseline":
        delta = r["r2"] - baseline_r2
        delta_str = f"{delta:+.6f}"
        bar = "+" * max(1, int(delta * 10000)) if delta > 0 else ""
        print(f"{r['label']:<40} {r2_str:>10}  {delta_str:>10}  {bar}")
    else:
        print(f"{r['label']:<40} {r2_str:>10}  {'-':>10}")

# 单独分析各组件贡献
print(f"\n组件贡献拆解:")
regime_only = results["ablation_b_regime"]["r2"]
ricci_only = results["ablation_c_ricci"]["r2"]
both = results["ablation_d_both"]["r2"]

if all(v is not None for v in [baseline_r2, regime_only, ricci_only, both]):
    print(f"  torsion (Δ+regime) 独立贡献: {regime_only - baseline_r2:+.6f}")
    print(f"  Ricci 权重 独立贡献:        {ricci_only - baseline_r2:+.6f}")
    print(f"  两者叠加贡献:              {both - baseline_r2:+.6f}")
    combined = (regime_only - baseline_r2) + (ricci_only - baseline_r2)
    print(f"  独立贡献之和:              {combined:+.6f}")
    synergy = both - baseline_r2 - combined
    if synergy > 0:
        print(f"  协同效应 (1+1>2):          {synergy:+.6f}")
    elif synergy < 0:
        print(f"  部分重叠:                  {synergy:+.6f}")

elapsed = time.time() - t_start
print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# 保存报告
report = {
    "baseline_r2": baseline_r2,
    "results": {k: {"label": v["label"], "r2": v["r2"]} for k, v in results.items()},
}
report_path = Path("models") / "ablation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"报告已保存: {report_path}")
