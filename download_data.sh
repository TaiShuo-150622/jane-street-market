#!/bin/bash
# download_data.sh
# ================
# GPU 机器上运行：从 GitHub Release 下载数据分块并合并
# 用法: bash download_data.sh
# 依赖: wget (Linux 自带) 或 curl (macOS 自带)

set -e

RELEASE_TAG="v1.0"
BASE_URL="https://github.com/TaiShuo-150622/jane-street-market/releases/download/${RELEASE_TAG}"
CHUNK_DIR="data_chunks"
DATA_DIR="data/processed"
CHUNKS=(
    "train_processed.part00"
    "train_processed.part01"
    "train_processed.part02"
    "train_processed.part03"
    "train_processed.part04"
    "train_processed.part05"
    "checksums.md5"
    "merge.sh"
)

mkdir -p "$CHUNK_DIR"

# 选 wget 或 curl
if command -v wget &> /dev/null; then
    DL_CMD="wget -c -O"
elif command -v curl &> /dev/null; then
    DL_CMD="curl -L -o"
else
    echo "错误: 需要 wget 或 curl"; exit 1
fi

echo "=== 从 GitHub Release 下载数据 ==="
echo ""

for chunk in "${CHUNKS[@]}"; do
    url="${BASE_URL}/${chunk}"
    dest="${CHUNK_DIR}/${chunk}"

    echo "  下载 ${chunk}..."
    if [ "$DL_CMD" = "wget -c -O" ]; then
        wget -c -O "$dest" "$url"   # -c 自动续传，已完成则跳过
    else
        curl -C - -L -o "$dest" "$url"  # -C - 也支持续传
    fi
done

# 合并 + 校验
echo ""
echo "=== 合并 + 校验 ==="
cd "$CHUNK_DIR"
bash merge.sh
cd ..

echo ""
echo "✅ 数据就绪！"
ls -lh "$DATA_DIR/train_processed.parquet"
