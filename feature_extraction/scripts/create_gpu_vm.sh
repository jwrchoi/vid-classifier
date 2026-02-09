#!/bin/bash
# =============================================================================
# Create a GCE spot VM with T4 GPU for feature extraction.
#
# Prerequisites:
#   1. gcloud CLI authenticated
#   2. Compute Engine API enabled (already done)
#   3. GPUS_ALL_REGIONS quota >= 1 (see below)
#
# Quota request (if GPUS_ALL_REGIONS = 0):
#   Go to: https://console.cloud.google.com/iam-admin/quotas?project=vid-classifier
#   Filter for "gpus_all_regions", click Edit Quotas, request limit = 1.
#   Approval usually takes 1-15 minutes for small increases.
# =============================================================================

set -euo pipefail

PROJECT="vid-classifier"
ZONE="us-central1-a"
VM_NAME="feature-extraction-gpu"
MACHINE_TYPE="n1-standard-8"  # 8 vCPU, 30 GB RAM — headroom for EasyOCR + YOLO
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

echo "Creating spot VM: ${VM_NAME} (${MACHINE_TYPE} + ${GPU_COUNT}x ${GPU_TYPE})"
echo "  Project: ${PROJECT}"
echo "  Zone:    ${ZONE}"
echo ""

gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --metadata=install-nvidia-driver=True \
    --scopes=storage-full \
    --no-restart-on-failure

echo ""
echo "VM created. To SSH in:"
echo "  gcloud compute ssh ${VM_NAME} --project=${PROJECT} --zone=${ZONE}"
echo ""
echo "Once connected, run:"
echo "  bash setup_vm.sh"
