#!/bin/bash
# package_results.sh
# ================
# 训练完成后在 GPU 机器上运行，打包所有模型文件
# 用法: bash package_results.sh
# 输出: results_YYYYMMDD_HHMM.tar.gz

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)
ARCHIVE="results_${TIMESTAMP}.tar.gz"
MODELS_DIR="models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "错误: models/ 目录不存在，训练可能还没跑"
    exit 1
fi

echo "=== 打包训练结果 ==="

# 检查关键文件
FILES_TO_PACK=()
for f in \
    catboost_model.cbm \
    mlp_full.pth \
    mlp_tda.pth \
    xgboost_model.json \
    ensemble_results.json \
    training_report.json \
    catboost_results.json \
    mlp_full_results.json \
    mlp_tda_results.json \
    xgboost_results.json \
    val_y.npy \
    val_w.npy \
    val_sids.npy \
    ; do
    if [ -f "$MODELS_DIR/$f" ]; then
        FILES_TO_PACK+=("$MODELS_DIR/$f")
        size=$(ls -lh "$MODELS_DIR/$f" | awk '{print $5}')
        echo "  ✅ $f  ($size)"
    else
        echo "  ⚠ $f  (不存在，跳过)"
    fi
done

echo ""
echo "  打包到: $ARCHIVE"
tar czf "$ARCHIVE" "${FILES_TO_PACK[@]}"

SIZE=$(ls -lh "$ARCHIVE" | awk '{print $5}')
echo "  完成: $ARCHIVE ($SIZE)"
echo ""
echo "=== 发给 Taishuo ==="
echo ""
echo "  方式1 — GitHub Release:"
echo "    在 https://github.com/TaiShuo-150622/jane-street-market/releases/new"
echo "    创建新 Release (tag v1.1)，上传 $ARCHIVE"
echo ""
echo "  方式2 — scp 直传:"
echo "    scp $ARCHIVE taishuo@你的IP:~/"
echo ""
echo "  方式3 — 网盘:"
echo "    上传 $ARCHIVE 到网盘，分享链接"
