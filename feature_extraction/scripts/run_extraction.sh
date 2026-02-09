#!/bin/bash
# =============================================================================
# Run Feature Extraction — one extractor at a time for resilience.
#
# Why one at a time?
#   - If one extractor fails, others are not affected.
#   - Easier to monitor progress and restart a specific extractor.
#   - On spot VMs, checkpointing every 25 videos uploads CSVs to GCS,
#     so preemption only loses at most 25 videos of work per extractor.
#
# Each extractor auto-skips videos already in its output CSV,
# so this script is safe to re-run after a preemption or crash.
#
# Usage:
#   bash feature_extraction/scripts/run_extraction.sh
#
# To run only specific extractors:
#   bash feature_extraction/scripts/run_extraction.sh cuts density
# =============================================================================

set -euo pipefail

cd "$HOME/vid-classifier"
source venv/bin/activate

VIDEO_LIST="data/video_list_v2.csv"
CHECKPOINT_INTERVAL=25  # Upload to GCS every 25 videos

# If specific extractors are passed as arguments, use them.
# Otherwise, run all in the recommended order.
if [ $# -gt 0 ]; then
    EXTRACTORS=("$@")
else
    # Order: fastest first, so we get partial results quickly.
    # gaze (~2s) → cuts (~3s) → density (~7s) → object_detection (~6s) → text_detection (~37s)
    EXTRACTORS=("gaze" "cuts" "density" "object_detection" "text_detection")
fi

echo "============================================================"
echo "  Feature Extraction Pipeline"
echo "  Extractors: ${EXTRACTORS[*]}"
echo "  Checkpoint: every ${CHECKPOINT_INTERVAL} videos → GCS"
echo "============================================================"
echo ""

for extractor in "${EXTRACTORS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Starting: ${extractor}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    python3 -m feature_extraction.extract_all \
        --extractors "${extractor}" \
        --video-list "${VIDEO_LIST}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        2>&1 | tee -a "data/features/${extractor}.log"

    echo ""
    echo "  Finished: ${extractor} at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================================"
echo "  All extractors complete."
echo "  Results in: data/features/"
echo "  Also uploaded to: gs://vid-classifier-db/features/"
echo "============================================================"

# Final disk usage report.
echo ""
echo "Disk usage:"
df -h / | tail -1
echo "Feature CSV sizes:"
ls -lh data/features/*.csv 2>/dev/null || echo "  (no CSVs found)"
