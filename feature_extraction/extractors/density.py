"""
Visual Density / Complexity Extractor
=======================================

Measures multiple dimensions of visual "complexity" in a video by analyzing
sampled frames.  These metrics help quantify how visually busy or stimulating
a TikTok video is.

WHY THIS MATTERS FOR RESEARCH
------------------------------
Visual density correlates with viewer attention and arousal.  Product-review
TikToks with lots of text overlays, fast motion, and many on-screen objects
tend to have higher density scores than simple talking-head vlogs.

HOW IT WORKS (COMPUTER VISION CONCEPTS)
-----------------------------------------

1. **Color Entropy** (scipy + OpenCV)
   - Convert each frame to HSV color space (Hue, Saturation, Value).
   - Compute a normalized histogram of pixel values per channel.
   - Compute Shannon entropy of each histogram:
       H = -sum(p * log2(p))   for each bin probability p
   - High entropy = many colors spread evenly (visually complex).
   - Low entropy = few dominant colors (simple background, solid color).
   - We average the three channel entropies into a single score.

2. **Edge Density** (Canny edge detector, OpenCV)
   - Convert frame to grayscale.
   - Apply Canny edge detection, which finds pixels where brightness
     changes sharply (edges of objects, text, patterns).
   - edge_density_ratio = (# edge pixels) / (total pixels).
   - High ratio = lots of detail, text, or texture.
   - Low ratio = smooth surfaces, solid colors.

3. **Motion Magnitude** (Farneback dense optical flow, OpenCV)
   - Compare consecutive sampled frames using dense optical flow.
   - Optical flow estimates a 2D motion vector (dx, dy) for every pixel,
     telling us how far and in which direction each pixel "moved" between
     the two frames.
   - We compute the magnitude sqrt(dx² + dy²) at each pixel and average
     across the frame.
   - High motion = fast camera movement, zooms, or subject movement.
   - Low motion = static camera, minimal subject movement.

OUTPUT COLUMNS
--------------
- video_id:               Unique video identifier
- color_entropy:          Average Shannon entropy of HSV histograms (higher = more colors)
- edge_density_ratio:     Average fraction of edge pixels per frame (higher = more detail)
- avg_motion_magnitude:   Average optical flow magnitude across frame pairs (higher = more motion)
- num_frames_sampled:     How many frames were analyzed

DEPENDENCIES
------------
- opencv-python   (cv2)
- scipy           (scipy.stats.entropy)
- numpy
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.stats import entropy as shannon_entropy

from feature_extraction.config import (
    FRAME_SAMPLE_INTERVAL,
    MAX_FRAMES_PER_VIDEO,
    COLOR_HISTOGRAM_BINS,
    CANNY_LOW,
    CANNY_HIGH,
)


# ---------------------------------------------------------------------------
# Per-frame feature functions
# ---------------------------------------------------------------------------

def compute_color_entropy(frame_bgr: np.ndarray, bins: int = COLOR_HISTOGRAM_BINS) -> float:
    """
    Compute average Shannon entropy of the HSV histogram.

    Shannon entropy measures the "surprise" or "information" in a distribution.
    A uniform histogram (all bins equal) has maximum entropy; a histogram with
    one dominant bin has low entropy.

    Args:
        frame_bgr: A single video frame in BGR format (as read by OpenCV).
        bins:      Number of histogram bins per channel.

    Returns:
        Average entropy across H, S, V channels (in bits).
    """
    # Convert BGR → HSV.  HSV separates color (H) from intensity (V),
    # making the entropy more perceptually meaningful than RGB.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    channel_entropies = []
    for ch in range(3):
        # Compute a normalized histogram (sums to 1.0 → probability distribution).
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / hist.sum()  # normalize to probabilities

        # Shannon entropy: H = -Σ p·log₂(p).
        # scipy.stats.entropy computes this (base=2 gives bits).
        channel_entropies.append(shannon_entropy(hist, base=2))

    return float(np.mean(channel_entropies))


def compute_edge_density(frame_bgr: np.ndarray, low: int = CANNY_LOW, high: int = CANNY_HIGH) -> float:
    """
    Compute the fraction of pixels that are edges (Canny detector).

    The Canny edge detector works in three steps:
      1. Smooth the image with a Gaussian filter (reduces noise).
      2. Compute gradient magnitude and direction at each pixel.
      3. Apply "non-maximum suppression" — only keep pixels where the gradient
         is a local maximum in the gradient direction (thin edges).
      4. Apply hysteresis thresholding with `low` and `high`:
         - Gradients > high → definitely an edge
         - Gradients between low and high → edge only if connected to a strong edge
         - Gradients < low → not an edge

    Args:
        frame_bgr: A single video frame in BGR format.
        low:       Lower hysteresis threshold.
        high:      Upper hysteresis threshold.

    Returns:
        Ratio of edge pixels to total pixels (0.0 to 1.0).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Canny returns a binary image: 255 for edge pixels, 0 for non-edge.
    edges = cv2.Canny(gray, low, high)

    # Count edge pixels (value 255) and divide by total pixels.
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = np.count_nonzero(edges)

    return edge_pixels / total_pixels


def compute_optical_flow_magnitude(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
) -> float:
    """
    Compute average motion magnitude between two frames using dense optical flow.

    Farneback optical flow estimates a 2D displacement vector (dx, dy) at every
    pixel.  The algorithm works by fitting local polynomial expansions to the
    image signal and solving for the displacement that minimizes the difference
    between the two expansions.

    The magnitude at each pixel is sqrt(dx² + dy²), measured in pixels of
    displacement.  We average this across the frame to get a single "how much
    stuff moved" score.

    Args:
        prev_bgr: Previous frame in BGR format.
        curr_bgr: Current frame in BGR format.

    Returns:
        Mean optical flow magnitude (pixels of displacement per frame pair).
    """
    # Optical flow works on grayscale images.
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    # calcOpticalFlowFarneback returns a (H, W, 2) array of (dx, dy) vectors.
    #   pyr_scale=0.5: each pyramid level is half the previous size
    #   levels=3:      number of pyramid levels (handles large motions)
    #   winsize=15:    averaging window size (larger = smoother but less local)
    #   iterations=3:  refinement iterations at each pyramid level
    #   poly_n=5:      pixel neighborhood for polynomial expansion
    #   poly_sigma=1.2: Gaussian sigma for polynomial smoothing
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Compute magnitude at each pixel: sqrt(dx² + dy²).
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    return float(magnitude.mean())


# ---------------------------------------------------------------------------
# Frame sampling helper
# ---------------------------------------------------------------------------

def sample_frames_bgr(
    video_path: Path,
    interval: int = FRAME_SAMPLE_INTERVAL,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> List[np.ndarray]:
    """
    Sample frames from a video at regular intervals, returning BGR arrays.

    Unlike shared.video_utils.sample_frames (which returns RGB), this keeps
    frames in BGR because OpenCV functions expect BGR input.

    Args:
        video_path: Path to a local video file.
        interval:   Sample every Nth frame.
        max_frames: Maximum number of frames to return.

    Returns:
        List of BGR numpy arrays.
    """
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(video_path: Path, video_id: str) -> Dict:
    """
    Full density-extraction pipeline for a single video.

    Samples frames, computes per-frame metrics, averages across the video.

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
            "color_entropy": None,
            "edge_density_ratio": None,
            "avg_motion_magnitude": None,
            "num_frames_sampled": 0,
        }

    # --- Per-frame metrics ---
    entropies = []
    edge_densities = []

    for frame in frames:
        entropies.append(compute_color_entropy(frame))
        edge_densities.append(compute_edge_density(frame))

    # --- Inter-frame motion ---
    motions = []
    for i in range(1, len(frames)):
        motions.append(compute_optical_flow_magnitude(frames[i - 1], frames[i]))

    return {
        "video_id": video_id,
        "color_entropy": round(float(np.mean(entropies)), 4),
        "edge_density_ratio": round(float(np.mean(edge_densities)), 6),
        "avg_motion_magnitude": round(float(np.mean(motions)), 4) if motions else 0.0,
        "num_frames_sampled": len(frames),
    }
