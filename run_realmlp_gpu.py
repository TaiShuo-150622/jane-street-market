"""RealMLP GPU — 采样跑，看 R² 再决定是否加量"""
import polars as pl, numpy as np, gc, time, json
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))
from src.data_utils import MODELS_DIR, TRAIN_PATH, TARGET_COL, WEIGHT_COL, TRAIN_END_DATE, ensure_models_dir
from src.metrics import weighted_r2
from pytabkit import RealMLP_TD_Regressor

ensure_models_dir()

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79) if i not in (9, 10, 11)]
CAT_FEATURE_COLS = ["feature_09", "feature_10", "feature_11"]
RESPONDER_COLS = [f"responder_{i}" for i in range(9) if i != 6]
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]

print("Loading...")
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

# 逐级尝试: 2M → 4M → 6M → 8M
for n_samples in [2_000_000, 4_000_000, 6_000_000, 8_000_000]:
    print(f"\n{'='*50}")
    print(f"Sampling {n_samples/1e6:.1f}M rows for GPU...")

    rng = np.random.RandomState(42)
    idx = rng.choice(train_df.height, min(n_samples, train_df.height), replace=False)

    X_tr = train_df[idx].select(avail).to_numpy().astype(np.float32)
    y_tr = train_df[idx][TARGET_COL].to_numpy().astype(np.float32).ravel()
    X_tr = np.nan_to_num(X_tr, nan=0.0); y_tr = np.nan_to_num(y_tr, nan=0.0)

    try:
        model = RealMLP_TD_Regressor(device='cuda', random_state=42, n_epochs=128, batch_size=256)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        pred = model.predict(X_val).astype(np.float64)
        r2 = weighted_r2(y_val, pred, w_val)
        print(f"  ✅ R²={r2:.6f}  ({elapsed:.0f}s)")

        del X_tr, y_tr, model; gc.collect()
    except Exception as e:
        print(f"  ❌ OOM: {str(e)[:100]}")
        break

print("\nDone!")
