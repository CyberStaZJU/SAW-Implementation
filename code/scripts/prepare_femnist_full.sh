#!/bin/bash
# ============================================================
# 下载并预处理完整的 FEMNIST 数据集（~3500 客户端）
# 使用 LEAF 项目的预处理流程
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${SAW_DATA_ROOT:-$HOME/SAW-data/data}"
STATE_ROOT="${SAW_STATE_ROOT:-$HOME/.local/state/SAW-Implementation}"
DATA_DIR="${SAW_FEMNIST_ROOT:-$DATA_ROOT/femnist}"
LEAF_DIR="${SAW_LEAF_ROOT:-$STATE_ROOT/deps/leaf}"
python3 - "$PROJECT_DIR/.." "$DATA_DIR" "$LEAF_DIR" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    if Path(value).expanduser().resolve().is_relative_to(source):
        raise SystemExit(f"Preparation path must be outside the repository: {value}")
PY
if [ -e "$DATA_DIR" ]; then
    echo "Existing FEMNIST output requires a fresh SAW_FEMNIST_ROOT: $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$(dirname "$DATA_DIR")" "$(dirname "$LEAF_DIR")"
mkdir "$DATA_DIR"

echo "=== FEMNIST Full Dataset Preparation ==="
echo "Project:  $PROJECT_DIR"
echo "Data dir: $DATA_DIR"
echo ""

# ---- Step 1: Clone LEAF repo (if not already) ----
if [ ! -d "$LEAF_DIR" ]; then
    echo "[1/5] Cloning LEAF repository..."
    git clone --depth=1 https://github.com/TalwalkarLab/leaf.git "$LEAF_DIR"
else
    echo "[1/5] LEAF repository already exists, skipping clone."
fi

# ---- Step 2: Install LEAF dependencies ----
echo "[2/5] Installing LEAF dependencies..."
cd "$LEAF_DIR"
pip install -q -r requirements.txt 2>/dev/null || true

# ---- Step 3: Download raw FEMNIST data ----
echo "[3/5] Downloading raw FEMNIST data from NIST..."
cd "$LEAF_DIR/data/femnist"

# Download raw NIST data if not already present
if [ ! -f "data/raw_data/by_write/by_write.hdf5" ] && [ ! -d "data/raw_data/by_class" ]; then
    echo "  Downloading by_class.zip (~350MB)..."
    mkdir -p data/raw_data
    cd data/raw_data
    wget -q --show-progress https://s3.amazonaws.com/nist-srd/SD19/by_class.zip || \
        wget -q --show-progress https://nist-srd.s3.amazonaws.com/SD19/by_class.zip
    echo "  Extracting by_class.zip..."
    unzip -q by_class.zip
    rm by_class.zip
    cd "$LEAF_DIR/data/femnist"

    echo "  Downloading by_write.zip (~350MB)..."
    mkdir -p data/raw_data
    cd data/raw_data
    wget -q --show-progress https://s3.amazonaws.com/nist-srd/SD19/by_write.zip || \
        wget -q --show-progress https://nist-srd.s3.amazonaws.com/SD19/by_write.zip
    echo "  Extracting by_write.zip..."
    unzip -q by_write.zip
    rm by_write.zip
else
    echo "  Raw data already downloaded."
fi

# ---- Step 4: Preprocess FEMNIST ----
echo "[4/5] Preprocessing FEMNIST (this may take 10-20 minutes)..."
cd "$LEAF_DIR/data/femnist"
# -s niid: non-iid split by writer
# --sf 1.0: use 100% of data
# -k 0: minimum 0 samples per user
# -t user: partition by user (writer)
./preprocess.sh -s niid --sf 1.0 -k 0 -t user

# ---- Step 5: Copy output to SAW-data ----
echo "[5/5] Copying output to $DATA_DIR..."
mkdir -p "$DATA_DIR/train" "$DATA_DIR/test"

# Find the generated output directory
OUTPUT_DIR=$(ls -dt "$LEAF_DIR/data/femnist/data/train" 2>/dev/null | head -1 || echo "")

# The preprocess script typically places output in data/train/ and data/test/
# with names like all_data_<i>_niid_05_keep_0_train_<digits>.json

SRC_TRAIN="$LEAF_DIR/data/femnist/data/train"
SRC_TEST="$LEAF_DIR/data/femnist/data/test"

    if [ -d "$SRC_TRAIN" ] && [ "$(ls -A "$SRC_TRAIN"/*.json 2>/dev/null)" ]; then
    echo "  Copying new train JSON files..."
    cp "$SRC_TRAIN/"*.json "$DATA_DIR/train/"
    echo "  Done."
else
    echo "  WARNING: No new train data generated."
fi
if [ -d "$SRC_TEST" ] && [ "$(ls -A "$SRC_TEST"/*.json 2>/dev/null)" ]; then
    echo "  Copying new test JSON files..."
    cp "$SRC_TEST/"*.json "$DATA_DIR/test/"
    echo "  Done."
else
    echo "  WARNING: No new test data generated."
fi

# ---- Summary ----
echo ""
echo "=== Done! ==="
echo "Train files: $(ls "$DATA_DIR/train/"*.json 2>/dev/null | wc -l)"
echo "Test files:  $(ls "$DATA_DIR/test/"*.json 2>/dev/null | wc -l)"
du -sh "$DATA_DIR/train" "$DATA_DIR/test"

# Count clients
python3 -c "
import json, os
d = '$DATA_DIR/train'
total = 0
samples = 0
for f in sorted(os.listdir(d)):
    if f.endswith('.json'):
        with open(os.path.join(d, f)) as fh:
            data = json.load(fh)
        users = len(data['user_data'])
        total += users
        s = sum(len(v['y']) for v in data['user_data'].values())
        samples += s
        print(f'  {f}: {users} users, {s} samples')
print(f'  ---')
print(f'  TOTAL: {total} users, {samples} samples')
"

echo ""
echo "Ready for use: python main_fed.py --dataset femnist --num_users 3500 ..."
