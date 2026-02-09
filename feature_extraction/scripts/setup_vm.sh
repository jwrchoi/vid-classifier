#!/bin/bash
# =============================================================================
# VM Setup Script — run this once after SSHing into the GPU VM.
#
# What it does:
#   1. Verifies GPU / CUDA are working
#   2. Clones the repo
#   3. Creates a venv and installs dependencies
#   4. Downloads the video list from GCS
#   5. Runs a quick smoke test on 1 video
#
# Usage:
#   gcloud compute ssh feature-extraction-gpu --project=vid-classifier --zone=us-central1-a
#   # Then on the VM:
#   bash setup_vm.sh
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/jwrchoi/vid-classifier.git"
REPO_DIR="$HOME/vid-classifier"
GCS_BUCKET="vid-classifier-db"

echo "============================================================"
echo "  GPU VM Setup for Feature Extraction"
echo "============================================================"

# --- 1. Verify GPU ---
echo ""
echo "[1/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU driver may still be installing."
    echo "Wait 2-3 minutes and re-run this script."
    exit 1
fi

# --- 2. Clone repo ---
echo ""
echo "[2/5] Cloning repository..."
if [ -d "${REPO_DIR}" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "${REPO_DIR}" && git pull
else
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

# --- 3. Create venv and install deps ---
echo ""
echo "[3/5] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r feature_extraction/requirements.txt --quiet

# Verify CUDA is visible to PyTorch.
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  PyTorch CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available to PyTorch. GPU acceleration disabled.')
    print('  EasyOCR and YOLO will fall back to CPU (slower but functional).')
"

# --- 4. Download video list and any existing feature CSVs from GCS ---
echo ""
echo "[4/5] Downloading data from GCS..."
mkdir -p data/features

# Video list.
gsutil cp "gs://${GCS_BUCKET}/annotations/video_list_v2.csv" data/video_list_v2.csv
echo "  Video list: $(wc -l < data/video_list_v2.csv) rows"

# Pull any existing feature CSVs (resume from previous runs).
gsutil -m cp "gs://${GCS_BUCKET}/features/*.csv" data/features/ 2>/dev/null || \
    echo "  No existing feature CSVs in GCS (starting fresh)."

for f in data/features/*.csv; do
    [ -f "$f" ] && echo "  $(basename $f): $(wc -l < $f) rows"
done

# --- 5. Smoke test: run density on first video ---
echo ""
echo "[5/5] Smoke test (density on 1 video)..."
python3 -m feature_extraction.extract_all \
    --extractors density \
    --video-list data/video_list_v2.csv \
    --limit 1 \
    --no-skip

echo ""
echo "============================================================"
echo "  Setup complete! Run extraction with:"
echo "    bash feature_extraction/scripts/run_extraction.sh"
echo "============================================================"
