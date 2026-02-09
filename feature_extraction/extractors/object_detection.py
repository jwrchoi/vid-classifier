"""
Object Detection Extractor
============================

Uses YOLOv8 to detect and count objects in sampled video frames.

WHY THIS MATTERS FOR RESEARCH
------------------------------
The number and types of objects on screen contribute to visual density.
For parasocial interaction research, the presence and count of **people**
is especially important — a single face close-up (personal distance) vs.
multiple people in a wide shot (public distance) are very different.

HOW IT WORKS (COMPUTER VISION CONCEPTS)
-----------------------------------------

**YOLO** (You Only Look Once) is a real-time object detection model.
Unlike older approaches that slide a window across the image, YOLO
processes the entire image in one pass ("one look"):

1. The image is divided into a grid of cells.
2. Each cell predicts bounding boxes and class probabilities simultaneously.
3. Non-maximum suppression (NMS) removes duplicate overlapping detections.

**YOLOv8** (by Ultralytics) is the latest version, pre-trained on the
COCO dataset with 80 object classes:
  person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
  traffic light, fire hydrant, stop sign, ..., cell phone, book, clock,
  scissors, teddy bear, toothbrush

We use the **nano** variant (yolov8n) which is fast enough for batch
processing TikTok videos while still being accurate for counting.

OUTPUT COLUMNS
--------------
- video_id:      Unique video identifier
- avg_objects:   Average number of detected objects per frame
- num_humans:    Average number of "person" detections per frame
- object_types:  JSON list of unique COCO class names seen across all frames
- num_frames_sampled: How many frames were analyzed

DEPENDENCIES
------------
- ultralytics    pip install ultralytics
- opencv-python
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from feature_extraction.config import (
    FRAME_SAMPLE_INTERVAL,
    MAX_FRAMES_PER_VIDEO,
    YOLO_MODEL_NAME,
    YOLO_CONFIDENCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Lazy-load the YOLO model so we only pay the cost once across many videos.
# The model is downloaded automatically on first use (~6 MB for nano).
# ---------------------------------------------------------------------------
_yolo_model = None


def _get_model():
    """Load YOLOv8 model (cached after first call)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_NAME)
    return _yolo_model


def detect_objects_in_frame(frame_bgr: np.ndarray) -> Dict:
    """
    Run YOLOv8 on a single frame and return detection summary.

    Args:
        frame_bgr: Single video frame in BGR format.

    Returns:
        Dict with keys:
        - num_objects: Total number of detected objects.
        - num_humans:  Number of "person" detections.
        - class_names: List of detected class names (with duplicates).
    """
    model = _get_model()

    # Run inference.  verbose=False suppresses per-frame logging.
    # conf sets the minimum confidence threshold.
    results = model(frame_bgr, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    # YOLOv8 returns a list of Results objects (one per image in the batch).
    # We only passed one image, so take results[0].
    result = results[0]

    # result.boxes contains all detections.
    #   .cls  — tensor of class indices (e.g. 0 = person, 2 = car)
    #   .conf — tensor of confidence scores
    # result.names maps class index → class name string.
    if result.boxes is None or len(result.boxes) == 0:
        return {"num_objects": 0, "num_humans": 0, "class_names": []}

    class_indices = result.boxes.cls.cpu().numpy().astype(int)
    class_names = [result.names[idx] for idx in class_indices]

    return {
        "num_objects": len(class_names),
        "num_humans": class_names.count("person"),
        "class_names": class_names,
    }


def sample_frames_bgr(
    video_path: Path,
    interval: int = FRAME_SAMPLE_INTERVAL,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> List[np.ndarray]:
    """Sample frames from a video at regular intervals (BGR format)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    frame_idx = 0
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()
    return frames


def extract(video_path: Path, video_id: str) -> Dict:
    """
    Full object-detection pipeline for a single video.

    Samples frames, runs YOLOv8 on each, and aggregates counts.

    Args:
        video_path: Path to a local video file.
        video_id:   Unique video identifier.

    Returns:
        Flat dictionary of features ready for CSV output.
    """
    frames = sample_frames_bgr(video_path)

    if not frames:
        return {
            "video_id": video_id,
            "avg_objects": None,
            "num_humans": None,
            "object_types": "[]",
            "num_frames_sampled": 0,
        }

    total_objects = []
    total_humans = []
    all_class_names: List[str] = []

    for frame in frames:
        result = detect_objects_in_frame(frame)
        total_objects.append(result["num_objects"])
        total_humans.append(result["num_humans"])
        all_class_names.extend(result["class_names"])

    # Unique object types across all frames (sorted for determinism).
    unique_types = sorted(set(all_class_names))

    return {
        "video_id": video_id,
        "avg_objects": round(float(np.mean(total_objects)), 2),
        "num_humans": round(float(np.mean(total_humans)), 2),
        "object_types": json.dumps(unique_types),
        "num_frames_sampled": len(frames),
    }
