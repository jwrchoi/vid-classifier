"""
Text Detection Extractor
==========================

Uses EasyOCR to detect on-screen text in sampled video frames and measures
how much text is present and how fast it changes.

WHY THIS MATTERS FOR RESEARCH
------------------------------
TikTok creators frequently overlay text on videos — captions, product names,
prices, calls to action.  The amount and pace of text overlays is a strong
signal for content style:
- Product reviews: heavy text overlays with specs, prices, comparisons
- Vlogs: minimal text, maybe a caption at the start
- Brand content: stylized text animations, fast-changing slogans

HOW IT WORKS (COMPUTER VISION CONCEPTS)
-----------------------------------------

**EasyOCR** is a deep-learning OCR library that combines two neural networks:

1. **Text Detection** (CRAFT model):
   - Scans the image for regions that look like text.
   - Outputs bounding boxes (polygons) around each text region.
   - Works by predicting "character affinity" — how likely neighboring
     pixels are to belong to the same character/word.

2. **Text Recognition** (CRNN model):
   - Takes each detected text region (cropped from the image).
   - Runs a Convolutional Recurrent Neural Network to read the characters.
   - Outputs the recognized text string and a confidence score.

We use the detection bounding boxes to compute spatial metrics (how much
of the frame is covered by text), and we compare the recognized text
across frames to measure text change pace.

OUTPUT COLUMNS
--------------
- video_id:                Unique video identifier
- has_text:                Boolean — any text detected in any frame?
- text_area_ratio:         Average fraction of frame area covered by text bounding boxes
- avg_text_regions:        Average number of text regions detected per frame
- text_changes_per_second: How often the on-screen text content changes
                           (measured by comparing OCR output between consecutive frames)
- num_frames_sampled:      How many frames were analyzed

DEPENDENCIES
------------
- easyocr        pip install easyocr
- opencv-python
- numpy
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

from feature_extraction.config import (
    FRAME_SAMPLE_INTERVAL,
    MAX_FRAMES_PER_VIDEO,
    EASYOCR_LANGUAGES,
    EASYOCR_CONFIDENCE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Lazy-load the EasyOCR reader.  First call downloads the model (~100 MB).
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_reader():
    """Load EasyOCR reader (cached after first call)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # gpu=True uses CUDA if available; falls back to CPU automatically.
        _ocr_reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=True)
    return _ocr_reader


def detect_text_in_frame(frame_bgr: np.ndarray) -> Dict:
    """
    Run EasyOCR on a single frame and return text detection metrics.

    Args:
        frame_bgr: Single video frame in BGR format.

    Returns:
        Dict with keys:
        - text_regions:  Number of detected text regions.
        - text_area:     Total area of text bounding boxes in pixels.
        - frame_area:    Total frame area in pixels (for computing ratio).
        - texts:         Set of recognized text strings (lowercased, for change detection).
    """
    reader = _get_reader()

    # Convert BGR → RGB (EasyOCR expects RGB input).
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # readtext returns a list of (bbox, text, confidence) tuples.
    #   bbox:       list of 4 (x, y) corner points of the text region polygon
    #   text:       recognized text string
    #   confidence: float 0–1
    detections = reader.readtext(frame_rgb)

    frame_h, frame_w = frame_bgr.shape[:2]
    frame_area = frame_h * frame_w

    text_area = 0
    texts: Set[str] = set()
    region_count = 0

    for bbox, text, confidence in detections:
        # Skip low-confidence detections (likely false positives from textures).
        if confidence < EASYOCR_CONFIDENCE_THRESHOLD:
            continue

        region_count += 1

        # bbox is a list of 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        # Compute the area of the bounding box using the shoelace formula
        # (works for any convex polygon).
        pts = np.array(bbox)
        # Simplified: use the axis-aligned bounding rect for speed.
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        text_area += (x_max - x_min) * (y_max - y_min)

        # Store lowercased text for change comparison.
        texts.add(text.strip().lower())

    return {
        "text_regions": region_count,
        "text_area": text_area,
        "frame_area": frame_area,
        "texts": texts,
    }


def sample_frames_bgr(
    video_path: Path,
    interval: int = FRAME_SAMPLE_INTERVAL,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> Tuple[List[np.ndarray], float]:
    """
    Sample frames and return them with the video's FPS.

    Returns:
        Tuple of (list of BGR frames, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
    return frames, fps


def extract(video_path: Path, video_id: str) -> Dict:
    """
    Full text-detection pipeline for a single video.

    Samples frames, runs EasyOCR on each, and aggregates text presence
    and change-pace metrics.

    Args:
        video_path: Path to a local video file.
        video_id:   Unique video identifier.

    Returns:
        Flat dictionary of features ready for CSV output.
    """
    frames, fps = sample_frames_bgr(video_path)

    if not frames:
        return {
            "video_id": video_id,
            "has_text": False,
            "text_area_ratio": None,
            "avg_text_regions": None,
            "text_changes_per_second": None,
            "num_frames_sampled": 0,
        }

    # --- Run OCR on each frame ---
    frame_results = []
    for frame in frames:
        frame_results.append(detect_text_in_frame(frame))

    # --- Aggregate text area ratio ---
    area_ratios = []
    region_counts = []
    any_text = False

    for r in frame_results:
        ratio = r["text_area"] / r["frame_area"] if r["frame_area"] > 0 else 0
        area_ratios.append(ratio)
        region_counts.append(r["text_regions"])
        if r["text_regions"] > 0:
            any_text = True

    # --- Compute text change pace ---
    # Compare the set of recognized texts between consecutive sampled frames.
    # A "change" occurs when the text content differs (new text appeared,
    # old text disappeared, or text was replaced).
    text_changes = 0
    for i in range(1, len(frame_results)):
        prev_texts = frame_results[i - 1]["texts"]
        curr_texts = frame_results[i]["texts"]
        # If the sets of text are different, count it as a change.
        if prev_texts != curr_texts and (prev_texts or curr_texts):
            text_changes += 1

    # Time span covered by the sampled frames.
    # interval frames apart, so time between consecutive samples = interval / fps.
    if len(frames) > 1 and fps > 0:
        time_span = (len(frames) - 1) * (FRAME_SAMPLE_INTERVAL / fps)
        text_changes_per_second = text_changes / time_span if time_span > 0 else 0
    else:
        text_changes_per_second = 0

    return {
        "video_id": video_id,
        "has_text": any_text,
        "text_area_ratio": round(float(np.mean(area_ratios)), 6),
        "avg_text_regions": round(float(np.mean(region_counts)), 2),
        "text_changes_per_second": round(text_changes_per_second, 4),
        "num_frames_sampled": len(frames),
    }
