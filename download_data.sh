#!/bin/bash
# download_data.sh
# ================
# GPU 机器上运行：从 GitHub Release 下载数据分块并合并
# 用法: bash download_data.sh <github_release_url>
# 例如: bash download_data.sh https://github.com/yourname/jane-street/releases/tag/v1.0

set -e

RELEASE_URL="${1:-}"
CHUNK_DIR="data_chunks"
DATA_DIR="data/processed"

if [ -z "$RELEASE_URL" ]; then
    echo "用法: bash download_data.sh <GitHub Release URL>"
    echo "例如: bash download_data.sh https://github.com/yourname/jane-street/releases/tag/v1.0"
    echo ""
    echo "如果你有 Kaggle API，也可以重新生成数据:"
    echo "  python src/preprocess.py  (会自动从 Kaggle 下载原始数据)"
    exit 1
fi

# 从 Release 页面 URL 提取 API URL
# https://github.com/USER/REPO/releases/tag/TAG → API

mkdir -p "$CHUNK_DIR"

echo "=== 从 GitHub Release 下载数据 ==="
echo "  Release: $RELEASE_URL"
echo ""

# 使用 gh CLI（推荐）
if command -v gh &> /dev/null; then
    echo "  使用 gh CLI 下载..."

    # 提取 tag
    TAG=$(echo "$RELEASE_URL" | grep -oP 'tag/\K.*' || echo "v1.0")

    gh release download "$TAG" \
        --dir "$CHUNK_DIR" \
        --pattern "train_processed.part*" \
        --pattern "checksums.md5" \
        --pattern "merge.sh" \
        --clobber

else
    echo "  gh CLI 未安装。手动下载步骤:"
    echo "  1. 浏览器打开: $RELEASE_URL"
    echo "  2. 下载所有 train_processed.part* 文件到 data_chunks/"
    echo "  3. 下载 checksums.md5 和 merge.sh 到 data_chunks/"
    echo ""
    echo "  或者安装 gh CLI:"
    echo "    Ubuntu: sudo apt install gh"
    echo "    Mac: brew install gh"
    exit 1
fi

# 合并 + 校验
echo ""
echo "=== 合并 + 校验 ==="
cd "$CHUNK_DIR"
bash merge.sh
cd ..

echo ""
echo "✅ 数据就绪！"
ls -lh "$DATA_DIR/train_processed.parquet"
