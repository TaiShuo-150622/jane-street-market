"""MLP 宽度/深度扫描 — 带完整输出和资源监控"""
import polars as pl, numpy as np, gc, time, json, os, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.data_utils import MODELS_DIR, TRAIN_PATH, TARGET_COL, WEIGHT_COL, TRAIN_END_DATE, ensure_models_dir
from src.metrics import weighted_r2
from pytabkit import RealMLP_TD_Regressor

ensure_models_dir()

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79) if i not in (9, 10, 11)]
CAT_FEATURE_COLS = ["feature_09", "feature_10", "feature_11"]
RESPONDER_COLS = [f"responder_{i}" for i in range(9) if i != 6]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]

def progress(msg):
    print(msg, flush=True)

def gpu_stats():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader",
            shell=True, timeout=5
        ).decode().strip()
        return out
    except:
        return "N/A"

try:
    cpu_count = os.cpu_count()
except:
    cpu_count = "?"

progress("=" * 70)
progress("MLP Architecture Sweep")
progress(f"CPU cores: {cpu_count}  |  GPU: {gpu_stats()}")
progress("=" * 70)

# ---- 加载数据(一次) ----
progress("Loading data...")
df = pl.scan_parquet(TRAIN_PATH)
train_df = df.filter(pl.col("date_id") <= TRAIN_END_DATE).collect()
val_df   = df.filter(pl.col("date_id") > TRAIN_END_DATE).collect()

avail = [c for c in FEATURE_COLS + CAT_FEATURE_COLS + RESPONDER_COLS + LAG_COLS
         if c in train_df.columns]

X_val = val_df.select(avail).to_numpy().astype(np.float32)
y_val = val_df[TARGET_COL].to_numpy().astype(np.float32).ravel()
w_val = val_df[WEIGHT_COL].to_numpy().astype(np.float32).ravel()
X_val = np.nan_to_num(X_val, nan=0.0); y_val = np.nan_to_num(y_val, nan=0.0)
del val_df; gc.collect()
progress(f"Val: {X_val.shape[0]:,} rows × {X_val.shape[1]} cols\n")

# ---- 配置列表：不同层数 + 宽度 ----
CONFIGS = [
    ("[256,256,256]",       [256, 256, 256],      128),
    ("[512,512,512]",       [512, 512, 512],      128),
    ("[1024,1024,1024]",    [1024, 1024, 1024],   128),
    ("[512,512]",           [512, 512],           128),
    ("[256,512,256]",       [256, 512, 256],      128),
    ("[1024,512]",          [1024, 512],          128),
    ("[256,256,256,256]",   [256, 256, 256, 256], 128),  # 4层
    ("[256,128,64]",        [256, 128, 64],       128),  # 漏斗型
    ("[512,512,512]_half",  [512, 512, 512],      64),   # epoch 减半
]

SAMPLE_SIZES = [2_000_000, 4_000_000, 8_000_000, 16_000_000]

RESULT_FILE = MODELS_DIR / "mlp_sweep_results.json"
all_results = []
if RESULT_FILE.exists():
    import json as _json
    with open(RESULT_FILE) as f:
        all_results = _json.load(f)
        progress(f"加载已有结果: {len(all_results)} 条")

for cfg_name, hidden, n_epochs in CONFIGS:
    for n_samples in SAMPLE_SIZES:
        progress(f"\n{'='*60}")
        progress(f"CFG={cfg_name}  |  采样={n_samples/1e6:.1f}M行")
        progress(f"{'='*60}")

        rng = np.random.RandomState(42)
        idx = rng.choice(train_df.height, min(n_samples, train_df.height), replace=False)
        X_tr = train_df[idx].select(avail).to_numpy().astype(np.float32)
        y_tr = train_df[idx][TARGET_COL].to_numpy().astype(np.float32).ravel()
        X_tr = np.nan_to_num(X_tr, nan=0.0); y_tr = np.nan_to_num(y_tr, nan=0.0)

        try:
            t0 = time.time()
            model = RealMLP_TD_Regressor(
                device='cuda', verbosity=0, random_state=42,
                n_epochs=n_epochs, batch_size=8192,
                hidden_sizes=hidden,
            )
            model.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            pred = model.predict(X_val).astype(np.float64)
            r2_w = weighted_r2(y_val, pred, w_val)
            r2_std = float(1 - np.sum((y_val-pred)**2) / np.sum((y_val-y_val.mean())**2))
            gpu = gpu_stats()

            progress(f"  ✅ W-R²={r2_w:.6f}  Std-R²={r2_std:.6f}  耗时={elapsed:.0f}s  GPU:{gpu}")

            all_results.append({
                "cfg": cfg_name, "n_samples": n_samples,
                "w_r2": float(r2_w), "std_r2": float(r2_std),
                "time_s": elapsed, "gpu": gpu,
            })
            # 每跑完一个立即保存
            with open(RESULT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            progress(f"  💾 已保存 ({len(all_results)}/{len(CONFIGS)*len(SAMPLE_SIZES)})")

            del X_tr, y_tr, model; gc.collect()

        except Exception as e:
            progress(f"  ❌ {type(e).__name__}: {str(e)[:150]}")
            all_results.append({
                "cfg": cfg_name, "n_samples": n_samples,
                "error": str(e)[:200]
            })
            break  # OOM → 这个 config 后面的 sample size 也跑不了

# ---- 汇总 ----
progress(f"\n{'='*70}")
progress("FINAL SUMMARY")
progress(f"{'='*70}")
progress(f"{'Config':<20} {'Samples':>8} {'W-R²':>10} {'Std-R²':>10} {'Time':>8} {'GPU'}")
progress("-" * 75)
for r in all_results:
    if "error" in r:
        progress(f"{r['cfg']:<20} {r['n_samples']/1e6:>7.1f}M  {'ERROR':>10}  {r['error'][:40]}")
    else:
        progress(f"{r['cfg']:<20} {r['n_samples']/1e6:>7.1f}M  {r['w_r2']:>10.6f}  {r['std_r2']:>10.6f}  {r['time_s']:>7.0f}s  {r['gpu']}")

progress(f"\nAll results in: {RESULT_FILE}")
progress("ALL DONE!")
