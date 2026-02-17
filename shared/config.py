"""
Shared configuration for the TikTok Video Analysis monorepo.
=============================================================

Central place for settings used across multiple sub-projects
(annotation dashboard, feature extraction, model training).
"""

from pathlib import Path

# =============================================================================
# GCS CONFIGURATION
# =============================================================================

# Google Cloud Storage bucket that holds videos, annotations, and features.
# The same bucket is used by all sub-projects.
GCS_BUCKET_NAME = "vid-classifier-db"

# Prefix (folder path) where the filtered TikTok video files live inside the bucket.
GCS_VIDEO_PREFIX = "videos/01_filtered/"

# Prefix where extracted feature CSVs are stored.
GCS_FEATURES_PREFIX = "features/"

# =============================================================================
# LOCAL DATA PATHS
# =============================================================================

# Root data directory (at the monorepo root level).
# On Cloud Run this is GCS FUSE-mounted at /app/data.
DATA_DIR = Path(__file__).parent.parent / "data"

# Video list CSVs
VIDEO_LIST_FILE = DATA_DIR / "video_list_v2.csv"

# Annotations output
ANNOTATIONS_FILE = DATA_DIR / "annotations.csv"

# Feature extraction outputs
FEATURES_DIR = DATA_DIR / "features"

# =============================================================================
# VIDEO SETTINGS
# =============================================================================

# File extensions recognized as video files.
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
