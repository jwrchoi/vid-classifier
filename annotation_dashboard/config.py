"""
Configuration settings for the Running Shoe Video Classifier.
=============================================================

This file contains all configurable settings for the annotation dashboard.
Update the paths below to match your local setup.

Key sections:
- PATH CONFIGURATION: Where videos and models are stored
- DEVICE CONFIGURATION: PyTorch device selection (CPU/GPU/MPS)
- MODEL CONFIGURATION: Which trained models to use
- VIDEO PROCESSING: Frame sampling settings
- ANNOTATION CONFIGURATION: What features coders will annotate
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

# =============================================================================
# PATH CONFIGURATION - UPDATE THESE TO MATCH YOUR SETUP
# =============================================================================

# Anchor paths relative to this file so they work regardless of CWD.
# _APP_DIR  = annotation_dashboard/          (where config.py lives)
# _REPO_ROOT = tiktok_video_analysis/        (parent of annotation_dashboard/)
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent

# Directory containing your video files
# This is where the TikTok videos to be annotated are stored
# The dashboard will list all videos in this directory
VIDEO_DIR = Path("/Volumes/T9/running_brand_videos/01_media/01_sample-videos")

# Directory containing your trained model weights (.pth files)
# These are the ResNet-50 models trained to predict perspective and distance
# Models are used internally but predictions are NOT shown to coders (to avoid bias)
MODELS_DIR = Path("/Users/roycechoi/proj/tiktok_analysis/codes/artifacts/model_exports/resnet50_models")

# Alternative paths to check if the primary MODELS_DIR doesn't exist
# The app will try each path in order and use the first one that exists
MODELS_DIR_CANDIDATES = [
    MODELS_DIR,  # Primary path (set above)
    _REPO_ROOT.parent / "tiktok_analysis" / "codes" / "artifacts" / "model_exports" / "resnet50_models",
    _APP_DIR / "models",
    _REPO_ROOT / "artifacts" / "model_exports" / "resnet50_models",
    Path.home() / "models" / "resnet50_models",
]

# Output directory for annotations and results
# This is where the CSV files with human annotations will be saved
# Created automatically if it doesn't exist
#
# Path resolution:
#   - Local monorepo: data/ lives at the repo root (one level above annotation_dashboard/)
#   - Cloud Run:      data/ is GCS FUSE-mounted as a sibling of app files (/app/data)
# We check the repo-root location first, then fall back to sibling.
if (_REPO_ROOT / "data").exists():
    OUTPUT_DIR = _REPO_ROOT / "data"
else:
    OUTPUT_DIR = _APP_DIR / "data"

ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.csv"

# =============================================================================
# GCS CONFIGURATION
# =============================================================================

# Google Cloud Storage bucket and prefix for video files
GCS_BUCKET_NAME = "vid-classifier-db"
GCS_VIDEO_PREFIX = "videos/01_filtered/"

# Fixed video list for reliability testing (all coders see same 50 videos).
# This CSV lives in the data directory alongside annotations.
VIDEO_LIST_FILE = OUTPUT_DIR / "video_list_v2.csv"

# Active-learning queue (written by model_training/active_learning.py).
# When this file exists the dashboard presents frames in AL-selected order
# instead of the default shuffled order.
QUEUE_CSV_PATH = OUTPUT_DIR / "queue.csv"

# Salt used to seed the per-coder shuffle when no AL queue exists.
QUEUE_SEED_SALT = "v1"

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device():
    """
    Detect the best available device for PyTorch inference.

    Priority order:
    1. MPS (Apple Silicon) - Fast on M1/M2/M3 Macs
    2. CUDA (NVIDIA GPU) - Fast on systems with NVIDIA GPUs
    3. CPU - Fallback, slower but always available

    Returns:
        torch.device or None: The selected compute device, or None if torch unavailable
    """
    if torch is None:
        return None
    # Check for Apple Silicon (M1/M2/M3 chips)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Check for NVIDIA GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # Fallback to CPU
    else:
        return torch.device("cpu")

# The device that will be used for model inference
# This is set once when the app starts
DEVICE = get_device()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model definitions for the trained ResNet-50 classifiers
#
# IMPORTANT: The 'classes' list order MUST match exactly how the model was trained!
# If the order is wrong, predictions will be swapped/incorrect.
#
# Each model config contains:
#   - num_classes: Number of output classes
#   - classes: List of class names in training order
#   - model_file: Filename of the .pth weights file
#   - display_name: Human-readable name for UI
#   - description: What this model predicts

MODEL_CONFIGS = {
    # -------------------------------------------------------------------------
    # V7 Multi-class Models (ACTIVE - these are what we use)
    # -------------------------------------------------------------------------

    # Perspective model - predicts camera point of view
    # 1st person: Camera shows viewer's POV (hands visible, no face of camera operator)
    # 2nd person: Subject directly addresses viewer (talking to camera, eye contact)
    # 3rd person: Camera is objective observer (documentary style, no direct address)
    'pov_multi': {
        'num_classes': 3,
        'classes': ['1st person', '2nd person', '3rd person'],
        'model_file': 'pov_resnet_50_v7.pth',
        'display_name': 'Perspective (POV)',
        'description': '1st person (POV), 2nd person (direct address), 3rd person (observer)'
    },

    # Social distance model - predicts camera proximity to subject
    # Personal: Close-up, face fills frame, intimate feeling
    # Social: Conversational distance, head-and-shoulders to waist-up
    # Public: Wide shot, full body or multiple people, formal/distant feeling
    'social_distance_multi': {
        'num_classes': 3,
        'classes': ['Personal', 'Social', 'Public'],
        'model_file': 'social_distance_resnet_50_v7.pth',
        'display_name': 'Social Distance',
        'description': 'Personal (close), Social (mid), Public (wide/crowd)'
    },

    # -------------------------------------------------------------------------
    # Legacy Binary Models (kept for reference, not active)
    # -------------------------------------------------------------------------

    'pov_binary': {
        'num_classes': 2,
        'classes': ['2nd_person', 'not_2nd_person'],
        'model_file': 'pov_resnet_50_binary_v1.pth',
        'display_name': 'Perspective (Binary)',
        'description': '2nd person (direct address) vs. other perspectives'
    },
    'social_distance_binary': {
        'num_classes': 2,
        'classes': ['not_personal', 'personal'],
        'model_file': 'social_distance_resnet_50_binary_v1.pth',
        'display_name': 'Distance (Binary)',
        'description': 'Personal (close-up) vs. not personal (mid/wide shot)'
    },
    'gaze': {
        'num_classes': 2,
        'classes': ['gaze at', 'gaze away'],
        'model_file': 'gaze_resnet_50_v7.pth',
        'display_name': 'Gaze Direction',
        'description': 'Looking at camera vs. looking away'
    }
}

# Which models to load and run for the annotation tool
# These models run in the background but their predictions are NOT shown to coders
# (to avoid anchoring bias in human annotations)
#
# Currently using v7 multi-class models for:
# - Perspective: 1st person, 2nd person, 3rd person
# - Distance: Personal, Social, Public
ACTIVE_MODELS = [
    'pov_multi',           # 3-way perspective classification
    'social_distance_multi' # 3-way distance classification
]

# =============================================================================
# VIDEO PROCESSING CONFIGURATION
# =============================================================================

# Frame sampling settings
# The app extracts frames from videos to run through the models
# These settings control how many frames are sampled

# Sample 1 frame every N frames from the video
# For a 30fps video, interval=15 means sampling at ~2fps
# Lower = more frames (slower, more accurate), Higher = fewer frames (faster)
FRAME_SAMPLE_INTERVAL = 15

# Maximum number of frames to sample per video
# Prevents memory issues on very long videos
# 100 frames at 2fps covers ~50 seconds of video
MAX_FRAMES_PER_VIDEO = 100

# Number of evenly-spaced frames to extract per video for frame-level annotation
FRAMES_PER_VIDEO = 10

# Video file extensions that the app will recognize
# Other files in the video directory will be ignored
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

# =============================================================================
# IMAGE PREPROCESSING (must match how models were trained)
# =============================================================================

# Input image size for ResNet-50
# All frames are resized to this before inference
IMG_SIZE = 224

# ImageNet normalization values
# These are standard for ResNet models pre-trained on ImageNet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# =============================================================================
# ANNOTATION CONFIGURATION
# =============================================================================

# Features that human coders will annotate
#
# Note: Model predictions are NOT shown to coders to avoid bias.
# The models run in the background and predictions are saved to CSV
# for later analysis (comparing model vs human labels).

ANNOTATION_FEATURES = {
    # Perspective (Point of View)
    # This is what the coder will see and select from
    'Perspective': {
        'key': 'perspective',  # Column name in CSV
        'options': ['1st person', '2nd person', '3rd person', 'NA'],
        'model_key': 'pov_multi',  # Which model predicts this (for saving)
        'help': '1st=POV/hands visible, 2nd=talking to camera, 3rd=observer, NA=unclear'
    },

    # Social Distance (Camera proximity)
    'Distance': {
        'key': 'distance',
        'options': ['Personal', 'Social', 'Public', 'NA'],
        'model_key': 'social_distance_multi',
        'help': 'Personal=close-up face, Social=conversational, Public=wide shot, NA=unclear'
    }
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Confidence thresholds for color coding model predictions
# (These are used internally, not shown to coders)
CONFIDENCE_HIGH = 0.85   # Green indicator
CONFIDENCE_MEDIUM = 0.65  # Yellow indicator
# Below MEDIUM = Red indicator

# Number of videos to show in the navigation queue
QUEUE_DISPLAY_SIZE = 10

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_models_dir():
    """
    Find the first existing models directory from the candidate list.

    This allows the app to work across different machines with different
    directory structures, as long as one of the candidate paths exists.

    Returns:
        Path: The first existing models directory, or MODELS_DIR as fallback
    """
    for candidate in MODELS_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    # Return default even if not exists (will error later with clear message)
    return MODELS_DIR


def find_video_dir():
    """
    Check if video directory exists and count videos.

    Returns:
        tuple: (Path or None, video count)
    """
    if VIDEO_DIR.exists():
        # Count MP4 files (most common format for TikTok videos)
        videos = list(VIDEO_DIR.glob("*.mp4"))
        if videos:
            return VIDEO_DIR, len(videos)
    return None, 0


def ensure_output_dir():
    """
    Create output directory if it doesn't exist.

    This is called when the app starts to make sure we have a place
    to save annotations.

    Returns:
        Path: The output directory path
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
