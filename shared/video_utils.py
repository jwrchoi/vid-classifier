"""
Video processing utilities shared across the monorepo.
======================================================

Common functions for reading video metadata, sampling frames,
and extracting video IDs from filenames.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_video_info(video_path: Path) -> Dict:
    """
    Extract metadata from a video file.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary with keys: path, filename, fps, frame_count,
        width, height, duration_sec, size_mb.
        On error, returns a dict with an 'error' key.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return {"error": f"File not found: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "path": str(video_path),
            "filename": video_path.name,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        }
    finally:
        cap.release()


def sample_frames(
    video_path: Path,
    interval: int = 15,
    max_frames: int = 100,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Sample frames from a video at regular intervals.

    Args:
        video_path: Path to video file.
        interval: Sample every Nth frame.
        max_frames: Maximum number of frames to return.
        start_frame: Frame index to start sampling from.
        end_frame: Frame index to stop at (None = end of video).

    Returns:
        Tuple of (list of RGB frames as numpy arrays, metadata dict).
    """
    video_path = Path(video_path)
    frames: List[np.ndarray] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"error": f"Could not open video: {video_path}"}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        sampled_indices: List[int] = []

        while frame_idx < end_frame and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % interval == 0:
                # OpenCV reads in BGR; convert to RGB for consistency.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                sampled_indices.append(frame_idx)

            frame_idx += 1

        metadata = {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "sampled_count": len(frames),
            "interval": interval,
            "sampled_indices": sampled_indices,
        }
        return frames, metadata

    finally:
        cap.release()


def extract_video_id(filename: str) -> str:
    """
    Extract video ID from a filename.

    Expected formats:
    - video-username-timestamp-videoid.mp4  ->  videoid
    - videoid.mp4                           ->  videoid

    Args:
        filename: Video filename (with or without directory prefix).

    Returns:
        Extracted video ID as a string.
    """
    name = Path(filename).stem

    # Try pattern: video-user-timestamp-id (last segment is the numeric ID).
    parts = name.split("-")
    if len(parts) >= 4:
        potential_id = parts[-1]
        if potential_id.isdigit() and len(potential_id) > 10:
            return potential_id

    return name
