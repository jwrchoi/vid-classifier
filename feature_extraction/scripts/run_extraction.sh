#!/bin/bash
# =============================================================================
# Run Feature Extraction — parallelized for speed.
#
# CPU extractors (gaze, cuts, density, object_detection) run in parallel
# with the GPU extractor (text_detection). Each extractor processes all
# videos independently, auto-skipping any already in its output CSV.
#
# Checkpointing every 25 videos uploads CSVs to GCS, so a crash loses
# at most 25 videos of work per extractor.
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

CHECKPOINT_INTERVAL=25  # Upload to GCS every 25 videos

# If specific extractors are passed as arguments, run them sequentially
# (custom runs don't parallelize — user controls the order).
if [ $# -gt 0 ]; then
    echo "============================================================"
    echo "  Feature Extraction Pipeline (custom extractors)"
    echo "  Extractors: $*"
    echo "  Checkpoint: every ${CHECKPOINT_INTERVAL} videos → GCS"
    echo "============================================================"

    for extractor in "$@"; do
        echo ""
        echo "  Starting: ${extractor} at $(date '+%Y-%m-%d %H:%M:%S')"
        python3 -m feature_extraction.extract_all \
            --extractors "${extractor}" \
            --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
            2>&1 | tee -a "data/features/${extractor}.log"
        echo "  Finished: ${extractor} at $(date '+%Y-%m-%d %H:%M:%S')"
    done
    exit 0
fi

# Default: run all extractors with parallelization.
# CPU extractors run in parallel with each other AND with text_detection (GPU).
CPU_EXTRACTORS=("gaze" "cuts" "density" "object_detection")
GPU_EXTRACTORS=("text_detection")

echo "============================================================"
echo "  Feature Extraction Pipeline (parallelized)"
echo "  CPU extractors: ${CPU_EXTRACTORS[*]} (parallel)"
echo "  GPU extractors: ${GPU_EXTRACTORS[*]} (parallel with CPU)"
echo "  Checkpoint: every ${CHECKPOINT_INTERVAL} videos → GCS"
echo "============================================================"
echo ""

PIDS=()

# Launch each CPU extractor as a background process.
for extractor in "${CPU_EXTRACTORS[@]}"; do
    echo "  Launching: ${extractor} (CPU) at $(date '+%Y-%m-%d %H:%M:%S')"
    python3 -m feature_extraction.extract_all \
        --extractors "${extractor}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        > "data/features/${extractor}.log" 2>&1 &
    PIDS+=($!)
done

# Launch GPU extractor in parallel.
for extractor in "${GPU_EXTRACTORS[@]}"; do
    echo "  Launching: ${extractor} (GPU) at $(date '+%Y-%m-%d %H:%M:%S')"
    python3 -m feature_extraction.extract_all \
        --extractors "${extractor}" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        > "data/features/${extractor}.log" 2>&1 &
    PIDS+=($!)
done

ALL_EXTRACTORS=("${CPU_EXTRACTORS[@]}" "${GPU_EXTRACTORS[@]}")

echo ""
echo "  All ${#PIDS[@]} extractors launched. PIDs: ${PIDS[*]}"
echo "  Logs: data/features/<extractor>.log"
echo ""
echo "  Monitor progress with:"
echo "    tail -f data/features/text_detection.log"
echo "    tail -f data/features/density.log"
echo ""

# Wait for all processes and track results.
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${ALL_EXTRACTORS[$i]}
    if wait "$pid"; then
        echo "  Completed: ${name} at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "  FAILED: ${name} (exit code $?) at $(date '+%Y-%m-%d %H:%M:%S')"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  All extractors finished. (${FAILED} failed)"
echo "  Results in: data/features/"
echo "  Also uploaded to: gs://vid-classifier-db/features/"
echo "============================================================"

# Final disk usage report.
echo ""
echo "Disk usage:"
df -h / | tail -1
echo "Feature CSV sizes:"
ls -lh data/features/*.csv 2>/dev/null || echo "  (no CSVs found)"
