# download_data.ps1
# ==================
# Windows PowerShell: 从 GitHub Release 下载数据分块并合并
# 用法: .\download_data.ps1
#
# 前提: 已经用 git clone 拉取了代码仓库

$ErrorActionPreference = "Stop"

$RELEASE_TAG = "v1.0"
$BASE_URL = "https://github.com/TaiShuo-150622/jane-street-market/releases/download/$RELEASE_TAG"
$CHUNK_DIR = "data_chunks"
$DATA_DIR = "data\processed"

$CHUNKS = @(
    "train_processed.part00",
    "train_processed.part01",
    "train_processed.part02",
    "train_processed.part03",
    "train_processed.part04",
    "train_processed.part05",
    "checksums.md5"
)

# 创建目录
if (-not (Test-Path $CHUNK_DIR)) {
    New-Item -ItemType Directory -Path $CHUNK_DIR | Out-Null
}

# 选择下载工具
$useCurl = $false
if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    $useCurl = $true
    Write-Host "使用 curl.exe 下载" -ForegroundColor Green
} else {
    Write-Host "使用 PowerShell Invoke-WebRequest 下载" -ForegroundColor Green
}

# ============================================================
# 1. 下载
# ============================================================
Write-Host "`n=== 从 GitHub Release 下载数据 ===" -ForegroundColor Cyan
Write-Host ""

foreach ($chunk in $CHUNKS) {
    $url = "$BASE_URL/$chunk"
    $dest = "$CHUNK_DIR\$chunk"

    # 跳过已存在且大小 > 0 的文件
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
        Write-Host "  跳过 (已存在): $chunk" -ForegroundColor Gray
        continue
    }

    Write-Host "  下载 $chunk ..." -ForegroundColor Yellow
    if ($useCurl) {
        # curl -C - 支持断点续传，-L 跟随重定向
        & curl.exe -L -C - -o "$dest" "$url"
    } else {
        Invoke-WebRequest -Uri $url -OutFile "$dest"
    }

    if ((Get-Item $dest).Length -eq 0) {
        Write-Host "  错误: 下载失败 ($chunk 大小为 0)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n  下载完成！" -ForegroundColor Green

# ============================================================
# 2. 合并
# ============================================================
Write-Host "`n=== 合并数据块 ===" -ForegroundColor Cyan

Push-Location $CHUNK_DIR

$outputFile = "train_processed.parquet"
$partFiles = Get-ChildItem "train_processed.part*" | Sort-Object Name

Write-Host "  合并 $($partFiles.Count) 个分块 → $outputFile ..." -ForegroundColor Yellow

# 使用 FileStream 合并二进制文件（比 Get-Content 快得多）
$outStream = [System.IO.File]::Create($outputFile)
$bufferSize = 4MB
$buffer = New-Object byte[] $bufferSize

foreach ($part in $partFiles) {
    Write-Host "    + $($part.Name) ($('{0:N1}' -f ($part.Length / 1GB)) GB)" -ForegroundColor Gray
    $inStream = [System.IO.File]::OpenRead($part.FullName)
    while ($true) {
        $read = $inStream.Read($buffer, 0, $bufferSize)
        if ($read -eq 0) { break }
        $outStream.Write($buffer, 0, $read)
    }
    $inStream.Close()
}
$outStream.Close()

$outputSize = (Get-Item $outputFile).Length
Write-Host "  合并完成: $outputFile ($('{0:N1}' -f ($outputSize / 1GB)) GB)" -ForegroundColor Green

# ============================================================
# 3. 校验 MD5
# ============================================================
Write-Host "`n=== 校验 MD5 ===" -ForegroundColor Cyan

$checksumFile = "checksums.md5"
if (-not (Test-Path $checksumFile)) {
    Write-Host "  错误: 找不到 $checksumFile" -ForegroundColor Red
    Pop-Location
    exit 1
}

$allOk = $true
Get-Content $checksumFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -match '^([a-f0-9]{32})\s+(.+)$') {
        $expectedHash = $matches[1]
        $fileName = $matches[2]

        if (-not (Test-Path $fileName)) {
            Write-Host "  MISS: $fileName" -ForegroundColor Red
            $allOk = $false
            return
        }

        $actualHash = (Get-FileHash -Algorithm MD5 $fileName).Hash.ToLower()
        if ($actualHash -eq $expectedHash) {
            Write-Host "  OK: $fileName" -ForegroundColor Green
        } else {
            Write-Host "  FAIL: $fileName" -ForegroundColor Red
            Write-Host "    期望: $expectedHash" -ForegroundColor Red
            Write-Host "    实际: $actualHash" -ForegroundColor Red
            $allOk = $false
        }
    }
}

if (-not $allOk) {
    Write-Host "`n  校验失败！请删除损坏的分块文件重新下载。" -ForegroundColor Red
    Pop-Location
    exit 1
}

# ============================================================
# 4. 移动到 data/processed/
# ============================================================
Write-Host "`n=== 移动到 data/processed/ ===" -ForegroundColor Cyan

$targetDir = "..\$DATA_DIR"
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

Move-Item -Force $outputFile $targetDir
Write-Host "  已移动: $DATA_DIR\train_processed.parquet" -ForegroundColor Green

Pop-Location

# ============================================================
# 完成
# ============================================================
Write-Host "`n=== 数据就绪！ ===" -ForegroundColor Cyan
Get-ChildItem "$DATA_DIR\train_processed.parquet" | ForEach-Object {
    Write-Host "  $($_.Name): $('{0:N1}' -f ($_.Length / 1GB)) GB" -ForegroundColor Green
}
Write-Host "`n下一步:" -ForegroundColor Cyan
Write-Host "  .\setup_gpu.ps1" -ForegroundColor White
Write-Host "  python check_env.py" -ForegroundColor White
Write-Host "  python train_all.py" -ForegroundColor White
