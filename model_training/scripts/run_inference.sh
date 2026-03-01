#!/usr/bin/env bash
# =============================================================================
# Run batch inference on the GCE T4 GPU VM.
#
# Steps:
#   1. Start the VM
#   2. Upload model weights
#   3. SSH in, pull repo, install deps, run inference
#   4. Download results
#   5. Stop the VM
#
# Usage:
#   bash model_training/scripts/run_inference.sh \
#       --pov-weights path/to/pov.pth \
#       --dist-weights path/to/dist.pth
# =============================================================================

set -euo pipefail

# ---- Config ----
VM_NAME="feature-extraction-gpu"
ZONE="us-east1-c"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
REPO_URL="https://github.com/roycechoi/tiktok_video_analysis.git"
REMOTE_DIR="/home/$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)/tiktok_video_analysis"

# ---- Parse args ----
POV_WEIGHTS=""
DIST_WEIGHTS=""
WORKERS=4
BATCH_SIZE=64

while [[ $# -gt 0 ]]; do
    case $1 in
        --pov-weights) POV_WEIGHTS="$2"; shift 2 ;;
        --dist-weights) DIST_WEIGHTS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$POV_WEIGHTS" || -z "$DIST_WEIGHTS" ]]; then
    echo "Usage: $0 --pov-weights <path> --dist-weights <path>"
    exit 1
fi

echo "============================================"
echo "  Batch Inference on GCE VM"
echo "  VM:     $VM_NAME ($ZONE)"
echo "  POV:    $POV_WEIGHTS"
echo "  Dist:   $DIST_WEIGHTS"
echo "============================================"

# ---- 1. Start VM ----
echo ""
echo "Starting VM..."
gcloud compute instances start "$VM_NAME" --zone="$ZONE" --quiet

echo "Waiting for VM to be ready..."
sleep 15

# ---- 2. Upload model weights ----
echo ""
echo "Uploading model weights..."
gcloud compute scp "$POV_WEIGHTS" "$VM_NAME:~/pov_model.pth" --zone="$ZONE" --quiet
gcloud compute scp "$DIST_WEIGHTS" "$VM_NAME:~/dist_model.pth" --zone="$ZONE" --quiet

# ---- 3. SSH in and run ----
echo ""
echo "Running inference on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet -- bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

echo "=== Setting up environment ==="

# Clone or pull repo
if [ -d ~/tiktok_video_analysis ]; then
    cd ~/tiktok_video_analysis && git pull
else
    cd ~ && git clone https://github.com/roycechoi/tiktok_video_analysis.git
    cd ~/tiktok_video_analysis
fi

# Install deps (use existing venv if available)
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet -r model_training/requirements.txt
pip install --quiet google-cloud-storage opencv-python-headless pandas numpy scikit-learn Pillow

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Running batch inference ==="
python3 -m model_training.batch_inference \
    --pov-weights ~/pov_model.pth \
    --dist-weights ~/dist_model.pth \
    --workers 4 \
    --batch-size 64 \
    --checkpoint-interval 500

echo ""
echo "=== Inference complete ==="
REMOTE_SCRIPT

# ---- 4. Download results ----
echo ""
echo "Downloading results..."
mkdir -p data/features

gcloud compute scp "$VM_NAME:~/tiktok_video_analysis/data/features/model_predictions.csv" \
    data/features/model_predictions.csv --zone="$ZONE" --quiet 2>/dev/null || \
    echo "  (frame CSV already uploaded to GCS by checkpoint)"

gcloud compute scp "$VM_NAME:~/tiktok_video_analysis/data/features/model_predictions_video.csv" \
    data/features/model_predictions_video.csv --zone="$ZONE" --quiet 2>/dev/null || \
    echo "  (video CSV already uploaded to GCS by checkpoint)"

# ---- 5. Stop VM ----
echo ""
echo "Stopping VM..."
gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --quiet

echo ""
echo "============================================"
echo "  Done! Results in data/features/"
echo "  - model_predictions.csv (frame-level)"
echo "  - model_predictions_video.csv (video-level)"
echo "============================================"
