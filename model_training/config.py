"""
Training configuration for the active-learning pipeline.

Imports shared constants and adds training-specific settings.
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parent

# Round outputs (models, metrics, queue CSVs)
ROUNDS_DIR = _TRAINING_DIR / "rounds"

# V8 fine-tuned model outputs
V8_OUTPUT_DIR = ROUNDS_DIR / "v8"

# Local frame PNG cache (large, gitignored)
FRAME_CACHE_DIR = _TRAINING_DIR / "frame_cache"

# Dashboard queue file — active_learning.py writes here so the dashboard
# picks up the AL-selected frame order on next startup.
DASHBOARD_QUEUE_PATH = _REPO_ROOT / "data" / "queue.csv"

# Annotations written by the dashboard
ANNOTATIONS_FILE = _REPO_ROOT / "data" / "annotations.csv"

# Video list CSV
VIDEO_LIST_FILE = _REPO_ROOT / "data" / "video_list_v2.csv"

# =============================================================================
# V7 PRETRAINED MODEL PATHS
# =============================================================================

# Candidate directories to find existing v7 model weights.
# Same search order as the dashboard config.
MODELS_DIR_CANDIDATES = [
    Path("/Users/roycechoi/proj/tiktok_analysis/codes/artifacts/model_exports/resnet50_models"),
    _REPO_ROOT.parent / "tiktok_analysis" / "codes" / "artifacts" / "model_exports" / "resnet50_models",
    _REPO_ROOT / "annotation_dashboard" / "models",
    _REPO_ROOT / "artifacts" / "model_exports" / "resnet50_models",
    Path.home() / "models" / "resnet50_models",
]


def find_models_dir() -> Path:
    """Return the first existing candidate models directory."""
    for candidate in MODELS_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODELS_DIR_CANDIDATES[0]


# Filenames inside the models directory
V7_POV_FILENAME = "pov_resnet_50_v7.pth"
V7_DISTANCE_FILENAME = "social_distance_resnet_50_v7.pth"

# =============================================================================
# MODEL / TASK CONFIGURATION
# =============================================================================

MODEL_CONFIGS = {
    "perspective": {
        "num_classes": 3,
        "classes": ["1st person", "2nd person", "3rd person"],
        "v7_filename": V7_POV_FILENAME,
        "output_filename": "pov_resnet50.pth",
    },
    "distance": {
        "num_classes": 3,
        "classes": ["Personal", "Social", "Public"],
        "v7_filename": V7_DISTANCE_FILENAME,
        "output_filename": "distance_resnet50.pth",
    },
}

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 5           # early stopping patience (epochs)
TEST_FRACTION = 0.2    # held-out test set size (created once in round 0)
TOP_K = 50             # frames selected per AL round

# =============================================================================
# IMAGE PREPROCESSING (must match inference / dashboard)
# =============================================================================

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# =============================================================================
# GCS
# =============================================================================

GCS_BUCKET_NAME = "vid-classifier-db"
FRAMES_PER_VIDEO = 10
