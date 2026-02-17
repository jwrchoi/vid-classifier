"""
Queue generation and management for randomized frame presentation.

Frames are shuffled across videos so coders see them in a random order
with no visible video grouping.  Each coder gets a deterministic but
unique order (seeded by annotator name + salt).

If an active-learning queue CSV exists (written by model_training/active_learning.py),
that queue takes priority.
"""

import hashlib
import random
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd


def generate_shuffled_queue(
    videos: list,
    frames_per_video: int,
    annotator: str,
    seed_salt: str = "v1",
) -> List[Tuple[str, int]]:
    """
    Create a shuffled list of (video_id, frame_index) tuples.

    The shuffle is reproducible: seeded by md5(annotator + seed_salt) so each
    coder gets a different but stable order.

    Args:
        videos: List of video info dicts (must have 'video_id').
        frames_per_video: Number of frames per video.
        annotator: Coder name.
        seed_salt: Additional salt for the hash seed.

    Returns:
        Shuffled list of (video_id, frame_index) pairs.
    """
    pairs = [
        (v["video_id"], fi)
        for v in videos
        for fi in range(frames_per_video)
    ]
    seed = int(hashlib.md5((annotator + seed_salt).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def load_queue_csv(queue_path: Path) -> List[Tuple[str, int]]:
    """
    Read a queue CSV produced by the active-learning pipeline.

    Expected columns: video_id, frame_index  (round and uncertainty_score optional).

    Returns:
        List of (video_id, frame_index) tuples in file order.
    """
    df = pd.read_csv(queue_path)
    df["video_id"] = df["video_id"].astype(str)
    df["frame_index"] = df["frame_index"].astype(int)
    return list(zip(df["video_id"], df["frame_index"]))


def get_effective_queue(
    videos: list,
    frames_per_video: int,
    annotator: str,
    queue_csv_path: Path,
    seed_salt: str = "v1",
) -> List[Tuple[str, int]]:
    """
    Return the frame queue to present.

    If *queue_csv_path* exists, load it (active-learning selected frames).
    Otherwise, generate a shuffled queue from the full video list.
    """
    if queue_csv_path.exists():
        return load_queue_csv(queue_csv_path)
    return generate_shuffled_queue(videos, frames_per_video, annotator, seed_salt)


def find_resume_position(
    queue: List[Tuple[str, int]],
    annotated_pairs: Set[Tuple[str, int]],
) -> int:
    """
    Return the index of the first unannotated item in *queue*.

    If everything is annotated, returns ``len(queue) - 1`` (last item).
    """
    for i, pair in enumerate(queue):
        if pair not in annotated_pairs:
            return i
    return max(len(queue) - 1, 0)
