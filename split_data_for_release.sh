#!/bin/bash
# split_data_for_release.sh
# ==========================
# 本地运行：把 11GB 数据切成 2GB 块 + 生成校验和
# 然后上传到 GitHub Release

set -e

DATA_FILE="data/processed/train_processed.parquet"
CHUNK_DIR="data_chunks"
CHUNK_SIZE="2000M"  # 2GB per chunk (GitHub Release 限制)

if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 找不到 $DATA_FILE"
    echo "请先运行 python src/preprocess.py 生成数据"
    exit 1
fi

echo "=== 数据分包 ==="
echo "  源文件: $DATA_FILE"
echo "  大小: $(ls -lh $DATA_FILE | awk '{print $5}')"
echo ""

# 清理旧的分包
rm -rf "$CHUNK_DIR"
mkdir -p "$CHUNK_DIR"

# 切分
echo "  切分成 ${CHUNK_SIZE} 块..."
split -b $CHUNK_SIZE "$DATA_FILE" "$CHUNK_DIR/chunk_"

# 重命名（加上序号方便辨识）
cd "$CHUNK_DIR"
i=0
for f in chunk_*; do
    new_name="train_processed.part$(printf '%02d' $i)"
    mv "$f" "$new_name"
    size=$(ls -lh "$new_name" | awk '{print $5}')
    echo "    $new_name  ($size)"
    ((i++))
done

# 生成校验和
echo ""
echo "  生成校验和..."
md5sum train_processed.part* > checksums.md5
echo "  完成！共 $(ls train_processed.part* | wc -l) 个分块"
echo ""

# 合并指令
cat > merge.sh << 'MERGE_EOF'
#!/bin/bash
# 在 GPU 机器上运行此脚本合并数据
echo "合并数据块..."
cat train_processed.part* > train_processed.parquet
echo "验证完整性..."
md5sum -c checksums.md5
if [ $? -eq 0 ]; then
    echo "✅ 数据合并完成 + 校验通过"
    echo "移动到 data/processed/"
    mkdir -p ../data/processed
    mv train_processed.parquet ../data/processed/
else
    echo "❌ 校验失败！请重新下载"
    exit 1
fi
MERGE_EOF
chmod +x merge.sh

cd ..
echo "=== 下一步 ==="
echo "1. 在 GitHub 上创建 Release (Tag: v1.0)"
echo "2. 把 data_chunks/ 下所有文件上传到 Release"
echo "3. 对方 clone 仓库后，下载 Release 文件放到 data_chunks/"
echo "4. 在 GPU 机器上运行: cd data_chunks && bash merge.sh"
