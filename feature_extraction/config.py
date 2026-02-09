"""
Configuration for the feature extraction pipeline.
====================================================

Controls which extractors to run, sampling parameters,
and output paths for feature CSVs.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the monorepo root importable so we can do `from shared import ...`
# ---------------------------------------------------------------------------
MONOREPO_ROOT = Path(__file__).resolve().parent.parent
if str(MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MONOREPO_ROOT))

from shared.config import (  # noqa: E402
    GCS_BUCKET_NAME,
    GCS_VIDEO_PREFIX,
    FEATURES_DIR,
    DATA_DIR,
    VIDEO_LIST_FILE,
)

# =============================================================================
# EXTRACTION SETTINGS
# =============================================================================

# How many frames to sample per video for frame-level features
# (object detection, text detection, gaze, etc.).
# For a 30 fps video, interval=15 means ~2 frames/sec.
FRAME_SAMPLE_INTERVAL = 15

# Maximum frames to process per video (prevents OOM on very long videos).
MAX_FRAMES_PER_VIDEO = 200

# =============================================================================
# CUT DETECTION (PySceneDetect)
# =============================================================================

# Minimum scene length in frames.  Scenes shorter than this are merged into
# the previous scene.  Prevents over-detection on quick flashes / effects.
MIN_SCENE_LENGTH_FRAMES = 10

# ContentDetector threshold — higher = fewer detected cuts.
# PySceneDetect's ContentDetector compares pixel-level changes between
# consecutive frames.  The default (27.0) works well for most content;
# TikTok videos with lots of text overlays may need a higher threshold
# to avoid false positives.
CONTENT_DETECTOR_THRESHOLD = 27.0

# =============================================================================
# OUTPUT
# =============================================================================

# Each extractor writes a CSV to this directory.
# CSVs are keyed on video_id for easy joining with annotations.
OUTPUT_DIR = FEATURES_DIR
