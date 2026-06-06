#!/usr/bin/env python3
"""
Smoke test - verify full pipeline with tiny data sample
Runs on CPU, ~2 min. Tests: data loading, all 3 models, checkpoint, ensemble.
"""

import sys, os, time, gc, json, shutil
from pathlib import Path
import numpy as np
import polars as pl

os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")

PASS = "OK"; FAIL = "FAIL"
passed = 0; failed = 0

TEST_N = 3000

def check(name, condition):
    global passed, failed
    if condition: print(f"  {PASS}  {name}"); passed += 1
    else: print(f"  {FAIL}  {name}"); failed += 1

print("=" * 60)
print("  Smoke Test - Full Pipeline Verification")
print("=" * 60)

# ---- 1. Imports ----
print("\n[1] Imports...")
try:
    from src.metrics import weighted_r2
    from src.data_utils import TRAIN_PATH, FULL_FEATURES_96, CAT_FEATURES, LAG_COLS, MODELS_DIR
    check("src.metrics", True); check("src.data_utils", True)
except Exception as e:
    check(f"imports: {e}", False); sys.exit(1)

# ---- 2. Data ----
print("\n[2] Data sampling...")
check("TRAIN_PATH exists", TRAIN_PATH.exists())

try:
    lazy = pl.scan_parquet(TRAIN_PATH)
    n_total = lazy.select(pl.len()).collect().item()
    print(f"  Total rows: {n_total:,}")

    # Train: first N rows
    train_sample = lazy.slice(0, TEST_N).collect()

    # Val: filter first (parquet predicate pushdown), then take sample
    val_sample = lazy.filter(pl.col("date_id") > 1400).slice(0, TEST_N).collect()
    if val_sample.height < 100:
        # Fallback: broader filter
        val_sample = lazy.filter(pl.col("date_id") > 1000).slice(0, TEST_N).collect()

    print(f"  Train sample: {train_sample.height} rows")
    print(f"  Val sample:   {val_sample.height} rows")
    check("train rows > 500", train_sample.height > 500)
    check("val rows > 100", val_sample.height > 100)

    if val_sample.height < 100:
        print("  WARNING: val sample too small, skipping val-dependent checks")
except Exception as e:
    check(f"data: {e}", False); import traceback; traceback.print_exc(); sys.exit(1)

# ---- 3. Metrics ----
print("\n[3] Metrics...")
y = np.array([1.0, 2.0, 3.0]); p = np.array([1.1, 1.9, 3.1]); w = np.ones(3)
check(f"weighted_r2={weighted_r2(y,p,w):.4f} (>0.9)", weighted_r2(y,p,w) > 0.9)
check("perfect R2=1.0", abs(weighted_r2(y,y,w) - 1.0) < 0.001)
check("zero pred R2=0", abs(weighted_r2(y, np.zeros(3), w)) < 0.001)

# Helper: polars to numpy (compatible with older polars)
def col_to_np(df, col, dtype=np.float64):
    arr = df[col].to_numpy()
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return np.nan_to_num(arr, nan=0.0)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- 4. CatBoost ----
print("\n[4] CatBoost (CPU, 20 iters)...")
try:
    from catboost import CatBoostRegressor, Pool

    feat_cols = FULL_FEATURES_96
    X_t = train_sample[feat_cols].to_pandas()
    y_t = col_to_np(train_sample, "responder_6")
    w_t = col_to_np(train_sample, "weight")
    X_v = val_sample[feat_cols].to_pandas()
    y_v = col_to_np(val_sample, "responder_6")
    w_v = col_to_np(val_sample, "weight")

    cat_idx = [feat_cols.index(c) for c in CAT_FEATURES if c in feat_cols]

    cb = CatBoostRegressor(
        iterations=20, learning_rate=0.1, depth=4,
        loss_function="RMSE", random_seed=42,
        allow_writing_files=False, silent=True,
    )
    cb.fit(Pool(X_t, y_t, weight=w_t, cat_features=cat_idx))
    pred = cb.predict(X_v)
    r2_cb = weighted_r2(y_v, pred, w_v)
    print(f"  Val R2={r2_cb:.6f}")
    check("CatBoost trained", True)

    # Save/load
    cb.save_model(str(MODELS_DIR / "_test_cb.cbm"))
    cb2 = CatBoostRegressor().load_model(str(MODELS_DIR / "_test_cb.cbm"))
    pred2 = cb2.predict(X_v)
    check("CatBoost save/load roundtrip", np.allclose(pred, pred2))
    (MODELS_DIR / "_test_cb.cbm").unlink()
    del cb, X_t, y_t
    gc.collect()
except Exception as e:
    check(f"CatBoost: {e}", False); import traceback; traceback.print_exc()

# ---- 5. XGBoost ----
print("\n[5] XGBoost (CPU, 20 trees)...")
try:
    import xgboost as xgb

    n_half = len(y_v) // 2
    X_v_np = val_sample[feat_cols].to_pandas().values.astype(np.float64)
    dtrain = xgb.DMatrix(X_v_np[:n_half], label=y_v[:n_half], weight=w_v[:n_half])
    dval = xgb.DMatrix(X_v_np[n_half:], label=y_v[n_half:], weight=w_v[n_half:])

    params = {"objective": "reg:squarederror", "learning_rate": 0.1, "max_depth": 4,
              "random_state": 42, "verbosity": 0}
    xgb_model = xgb.train(params, dtrain, num_boost_round=20, verbose_eval=False)

    pred_xgb = xgb_model.predict(dval)
    r2_xgb = weighted_r2(y_v[n_half:], pred_xgb, w_v[n_half:])
    print(f"  Val R2={r2_xgb:.6f}")
    check("XGBoost trained", True)

    xgb_model.save_model(str(MODELS_DIR / "_test_xgb.json"))
    xgb2 = xgb.Booster(); xgb2.load_model(str(MODELS_DIR / "_test_xgb.json"))
    pred2 = xgb2.predict(dval)
    check("XGBoost save/load roundtrip", np.allclose(pred_xgb, pred2, atol=1e-5))
    (MODELS_DIR / "_test_xgb.json").unlink()
    del dtrain, dval, xgb_model
    gc.collect()
except Exception as e:
    check(f"XGBoost: {e}", False); import traceback; traceback.print_exc()

# ---- 6. MLP ----
print("\n[6] MLP (CPU, 10 epochs)...")
try:
    import torch
    from src.train_mlp import MLP, JaneStreetDataset

    numeric_cols = [c for c in FULL_FEATURES_96 if c in train_sample.columns]
    X_t_np = train_sample[numeric_cols].to_pandas().values.astype(np.float32)
    y_t_np = col_to_np(train_sample, "responder_6", np.float32)
    w_t_np = col_to_np(train_sample, "weight", np.float32)

    mu = X_t_np.mean(axis=0); sg = X_t_np.std(axis=0); sg[sg==0]=1.0
    X_t_np = (X_t_np - mu) / sg

    X_v_np2 = val_sample[numeric_cols].to_pandas().values.astype(np.float32)
    X_v_np2 = (X_v_np2 - mu) / sg
    y_v_np = col_to_np(val_sample, "responder_6", np.float32)
    w_v_np = col_to_np(val_sample, "weight", np.float32)

    input_dim = X_t_np.shape[1]
    ds = JaneStreetDataset(X_t_np, y_t_np, w_t_np)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model = MLP(input_dim, [64, 32], [0.1, 0.1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    losses = []
    for ep in range(10):
        ep_loss = 0
        for bx, by, bw in loader:
            opt.zero_grad()
            loss = (torch.nn.functional.mse_loss(model(bx), by, reduction="none") * bw).mean()
            loss.backward(); opt.step()
            ep_loss += loss.item() * bx.size(0)
        losses.append(ep_loss / len(ds))
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    check("MLP loss decreasing", losses[-1] < losses[0])

    # Checkpoint
    torch.save({
        "epoch": 9, "model_state_dict": model.state_dict(),
        "hidden_dims": [64, 32], "input_dim": input_dim,
        "val_r2": 0.001, "history": {"train_loss": losses, "val_loss": [], "val_r2": []},
    }, str(MODELS_DIR / "_test_mlp.pth"))

    ckpt = torch.load(str(MODELS_DIR / "_test_mlp.pth"), weights_only=False)
    m2 = MLP(ckpt["input_dim"], ckpt["hidden_dims"], [0.1, 0.1])
    m2.load_state_dict(ckpt["model_state_dict"])
    check("MLP checkpoint roundtrip", True)
    (MODELS_DIR / "_test_mlp.pth").unlink()
    del model, ds, loader
    gc.collect()
except Exception as e:
    check(f"MLP: {e}", False); import traceback; traceback.print_exc()

# ---- 7. Ensemble ----
print("\n[7] Ensemble...")
try:
    from src.ensemble import load_and_ensemble
    np.random.seed(42)
    n = 300
    y_fake = np.random.randn(n).astype(np.float64)
    w_fake = np.ones(n)
    sids_fake = np.random.randint(0, 5, n)
    preds = {
        "a": y_fake * 0.8 + np.random.randn(n) * 0.4,
        "b": y_fake * 0.6 + np.random.randn(n) * 0.6,
        "c": y_fake * 0.7 + np.random.randn(n) * 0.5,
    }
    result = load_and_ensemble(preds, y_fake, w_fake, sids_fake, mode="grid_search", grid_step=0.1)
    check("Ensemble grid search done", "weights" in result)
    if "weights" in result:
        print(f"  Weights: {result['weights']}")
    best_ind = max(result.get("individual_scores", {}).values()) if result.get("individual_scores") else 0
    check("Ensemble R2 >= best individual", result["ensemble_r2"] >= best_ind - 0.01)
except Exception as e:
    check(f"Ensemble: {e}", False); import traceback; traceback.print_exc()

# ---- 8. Cleanup ----
print("\n[8] Cleanup...")
cache_dir = Path("models/mlp_cache")
if cache_dir.exists(): shutil.rmtree(cache_dir)
for f in Path("models").glob("_test_*"): f.unlink()
print("  Done")

# ---- Report ----
print(f"\n{'=' * 60}")
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print(f"  All passed! Ready for GPU machine.")
else:
    print(f"  {failed} failures - check above.")
print(f"{'=' * 60}")
sys.exit(0 if failed == 0 else 1)
