#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
PROJECT="vid-classifier"
REGION="us-central1"
SERVICE="shoe-annotator"
BUCKET="vid-classifier-db"
REPO="cloud-run-source-deploy"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}"
SA_NAME="${SERVICE}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "=== Deploying ${SERVICE} to Cloud Run ==="
echo "Project: ${PROJECT}  Region: ${REGION}  Bucket: ${BUCKET}"

# =============================================================================
# 1. Enable required APIs
# =============================================================================
echo "--- Enabling APIs ---"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT}"

# =============================================================================
# 2. Create Artifact Registry repo (if needed)
# =============================================================================
echo "--- Ensuring Artifact Registry repo ---"
gcloud artifacts repositories describe "${REPO}" \
    --location="${REGION}" --project="${PROJECT}" 2>/dev/null || \
gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT}"

# =============================================================================
# 3. Create service account (if needed)
# =============================================================================
echo "--- Ensuring service account ---"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
    --project="${PROJECT}" 2>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="Cloud Run ${SERVICE}" \
        --project="${PROJECT}"
    echo "Waiting for service account to propagate..."
    sleep 10
fi

# Grant Storage Object Admin on the bucket
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://${BUCKET}"

# =============================================================================
# 4. Upload seed data to GCS
# =============================================================================
# Resolve the monorepo data/ directory (one level above annotation_dashboard/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

echo "--- Uploading seed data to GCS ---"
gsutil cp "${DATA_DIR}/video_list_v2.csv" "gs://${BUCKET}/annotations/video_list_v2.csv"

# Upload annotations.csv only if it doesn't already exist in GCS
if ! gsutil stat "gs://${BUCKET}/annotations/annotations.csv" 2>/dev/null; then
    if [ -f "${DATA_DIR}/annotations.csv" ]; then
        gsutil cp "${DATA_DIR}/annotations.csv" "gs://${BUCKET}/annotations/annotations.csv"
    else
        echo "video_id,filename,annotator,perspective,distance,no_human_visible,notes,is_difficult,annotation_time_sec,timestamp" \
            | gsutil cp - "gs://${BUCKET}/annotations/annotations.csv"
    fi
    echo "Uploaded initial annotations.csv"
else
    echo "annotations.csv already exists in GCS, skipping"
fi

# =============================================================================
# 5. Build container image
# =============================================================================
echo "--- Building container image ---"
gcloud builds submit \
    --tag="${IMAGE}" \
    --project="${PROJECT}"

# =============================================================================
# 6. Deploy to Cloud Run with GCS FUSE volume mount
# =============================================================================
echo "--- Deploying to Cloud Run ---"
gcloud run deploy "${SERVICE}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --service-account="${SA_EMAIL}" \
    --allow-unauthenticated \
    --memory=1Gi \
    --execution-environment=gen2 \
    --add-volume=name=annotations-vol,type=cloud-storage,bucket="${BUCKET}",mount-options="only-dir=annotations" \
    --add-volume-mount=volume=annotations-vol,mount-path=/app/data

# =============================================================================
# 7. Print deployed URL
# =============================================================================
URL=$(gcloud run services describe "${SERVICE}" \
    --region="${REGION}" --project="${PROJECT}" \
    --format="value(status.url)")

echo ""
echo "=== Deployed successfully ==="
echo "URL: ${URL}"
