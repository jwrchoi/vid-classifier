"""
PyTorch Dataset for frame-level annotation data.

Loads frames by (video_id, frame_index).  Checks a local PNG cache first;
on miss, downloads the video from GCS, extracts all frames, and caches them.
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training.config import (
    FRAME_CACHE_DIR,
    FRAMES_PER_VIDEO,
    GCS_BUCKET_NAME,
    IMG_MEAN,
    IMG_SIZE,
    IMG_STD,
    VIDEO_LIST_FILE,
)


# =============================================================================
# FRAME CACHING
# =============================================================================

def _frame_cache_path(video_id: str, frame_index: int) -> Path:
    """Return local cache path for a single frame PNG."""
    return FRAME_CACHE_DIR / video_id / f"frame_{frame_index:03d}.png"


def _cache_video_frames(video_id: str, gcs_path: str) -> None:
    """Download a video from GCS, extract frames, and cache as PNGs."""
    out_dir = FRAME_CACHE_DIR / video_id
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= FRAMES_PER_VIDEO:
        return  # already cached

    from shared.gcs_utils import get_gcs_client

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    suffix = Path(gcs_path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        blob.download_to_filename(tmp.name)
        tmp.close()

        # Extract evenly-spaced frames (same logic as dashboard)
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return
        num_frames = FRAMES_PER_VIDEO
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]

        out_dir.mkdir(parents=True, exist_ok=True)
        for fi, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(str(out_dir / f"frame_{fi:03d}.png"))
        cap.release()
    finally:
        os.unlink(tmp.name)


def load_frame(video_id: str, frame_index: int, video_lookup: Dict[str, str]) -> Optional[Image.Image]:
    """
    Load a single frame as a PIL Image.

    Checks local cache first; on miss downloads + caches the whole video.

    Args:
        video_id: Video identifier.
        frame_index: 0-based frame index.
        video_lookup: Mapping video_id -> gcs_path.

    Returns:
        PIL Image (RGB) or None on failure.
    """
    path = _frame_cache_path(video_id, frame_index)
    if not path.exists():
        gcs_path = video_lookup.get(video_id)
        if gcs_path is None:
            return None
        _cache_video_frames(video_id, gcs_path)
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


# =============================================================================
# VIDEO LOOKUP
# =============================================================================

def build_video_lookup(video_list_path: Path = VIDEO_LIST_FILE) -> Dict[str, str]:
    """Return {video_id: gcs_path} from the video list CSV."""
    df = pd.read_csv(video_list_path)
    df["video_id"] = df["video_id"].astype(str)
    return dict(zip(df["video_id"], df["gcs_path"]))


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])


# =============================================================================
# DATASET
# =============================================================================

class FrameDataset(Dataset):
    """
    PyTorch Dataset that yields (image_tensor, label_index) for annotated frames.

    Filters to rows where the label is in the valid class list (excludes NA).
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        task: str,
        classes: List[str],
        video_lookup: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            annotations: DataFrame with columns video_id, frame_index, <task>.
            task: Column name ('perspective' or 'distance').
            classes: Ordered class list (index = label integer).
            video_lookup: {video_id: gcs_path}.
            transform: torchvision transform to apply.
        """
        self.task = task
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.video_lookup = video_lookup
        self.transform = transform or get_eval_transform()

        # Filter to valid labels
        df = annotations.copy()
        df["video_id"] = df["video_id"].astype(str)
        df = df[df[task].isin(classes)].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        frame_index = int(row["frame_index"])
        label = self.class_to_idx[row[self.task]]

        img = load_frame(video_id, frame_index, self.video_lookup)
        if img is None:
            # Return a blank image as fallback (should be rare)
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        tensor = self.transform(img)
        return tensor, label
