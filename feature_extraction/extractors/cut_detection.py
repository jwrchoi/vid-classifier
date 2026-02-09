"""
Cut / Scene-Change Detection
=============================

Detects "cuts" (hard transitions between scenes) in a video and computes
summary statistics about editing pace.

WHY THIS MATTERS FOR RESEARCH
------------------------------
Cut pace (how frequently the camera angle or scene changes) is an indicator
of editing style and viewer engagement strategies.  Fast cuts are common in
product-review TikToks; slow, unbroken shots are more common in personal
vlogs.  These metrics feed into the "density" dimension of parasocial
interaction analysis.

HOW IT WORKS (COMPUTER VISION CONCEPTS)
-----------------------------------------
We use PySceneDetect's **ContentDetector**, which works like this:

1. For each pair of consecutive frames (frame N and frame N+1), the detector
   computes a "content value" — a measure of how different the two frames are.

2. The content value is based on *per-channel mean pixel intensity differences*
   in the HSV color space:
     - H (hue): the "color" — e.g. red vs blue
     - S (saturation): how vivid the color is
     - V (value): brightness
   A large change in any of these channels suggests the scene has changed.

3. If the content value exceeds a **threshold** (default 27.0), the detector
   declares a cut at that frame boundary.

4. A **minimum scene length** filter prevents rapid flicker (e.g. a camera
   flash) from being counted as a real scene change.

OUTPUT COLUMNS
--------------
- video_id:            Unique video identifier (for joining with annotations)
- cut_count:           Total number of detected scene changes
- cuts_per_second:     cut_count / video_duration — the "editing pace" metric
- avg_scene_duration:  Average length of each scene in seconds
- min_scene_duration:  Shortest scene in seconds
- max_scene_duration:  Longest scene in seconds
- scene_timestamps:    JSON list of (start_sec, end_sec) tuples per scene

DEPENDENCIES
------------
- scenedetect (PySceneDetect)   pip install scenedetect[opencv]
- opencv-python                 pip install opencv-python
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

# ---------------------------------------------------------------------------
# PySceneDetect imports.
#
# SceneDetect has two main detection algorithms:
#   - ContentDetector: compares pixel-level content between frames (what we use)
#   - ThresholdDetector: detects fade-to-black transitions
#
# We use ContentDetector because TikTok videos rarely use fade-to-black;
# they typically hard-cut between scenes.
# ---------------------------------------------------------------------------
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Import config — this also sets up sys.path so `shared` is importable.
from feature_extraction.config import (
    MIN_SCENE_LENGTH_FRAMES,
    CONTENT_DETECTOR_THRESHOLD,
)


def detect_scenes(
    video_path: Path,
    threshold: float = CONTENT_DETECTOR_THRESHOLD,
    min_scene_len: int = MIN_SCENE_LENGTH_FRAMES,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Run scene detection on a single video file.

    This is the core detection function.  It opens the video, feeds every
    frame through the ContentDetector, and returns a list of scene boundaries.

    Args:
        video_path:    Path to a local video file.
        threshold:     ContentDetector sensitivity.  Higher = fewer cuts detected.
                       The detector fires when the frame-to-frame content change
                       score exceeds this value.  27.0 is a good default; raise to
                       ~35 if you get false positives on text-overlay transitions.
        min_scene_len: Minimum number of frames for a scene.  Scenes shorter than
                       this are merged into the previous scene.  Helps filter out
                       camera flashes and single-frame glitches.

    Returns:
        A tuple of:
        - scenes: List of (start_seconds, end_seconds) for each detected scene.
        - duration: Total video duration in seconds.
        - fps: Frames per second of the video.

    Example:
        >>> scenes, dur, fps = detect_scenes(Path("video.mp4"))
        >>> print(f"{len(scenes)} scenes in {dur:.1f}s video")
        5 scenes in 28.3s video
    """
    # -----------------------------------------------------------------------
    # 1. Open the video using PySceneDetect's video manager.
    #    This is a thin wrapper around OpenCV's VideoCapture that adds
    #    frame-accurate seeking and timecode support.
    # -----------------------------------------------------------------------
    video = open_video(str(video_path))

    # -----------------------------------------------------------------------
    # 2. Create a SceneManager and register our detector.
    #    The SceneManager coordinates the detection loop: it reads frames
    #    from the video and passes them to each registered detector.
    # -----------------------------------------------------------------------
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
        )
    )

    # -----------------------------------------------------------------------
    # 3. Run detection.  This reads every frame of the video (no skipping),
    #    because the detector needs consecutive frames to compare.
    #    For a 30-second TikTok at 30 fps, that's ~900 frames — fast enough.
    # -----------------------------------------------------------------------
    scene_manager.detect_scenes(video)

    # -----------------------------------------------------------------------
    # 4. Retrieve the detected scene list.
    #    Each scene is a tuple of (FrameTimecode_start, FrameTimecode_end).
    #    FrameTimecode objects can be converted to seconds via .get_seconds().
    # -----------------------------------------------------------------------
    scene_list = scene_manager.get_scene_list()

    # Get video metadata from OpenCV for duration / fps.
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback to 30 if unknown
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    # Convert FrameTimecode objects to plain (start_sec, end_sec) tuples.
    scenes = [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_list
    ]

    return scenes, duration, fps


def compute_cut_features(
    scenes: List[Tuple[float, float]],
    duration: float,
) -> Dict:
    """
    Compute summary statistics from the detected scene list.

    This is a pure-Python function (no CV dependencies) that takes the raw
    scene boundaries and turns them into meaningful research metrics.

    Args:
        scenes:   List of (start_sec, end_sec) tuples from detect_scenes().
        duration: Total video duration in seconds.

    Returns:
        Dictionary of cut-pace features.
    """
    # Number of cuts = number of scene boundaries - 1.
    # If the detector finds 5 scenes, there were 4 cuts between them.
    # Edge case: a single-scene video has 0 cuts.
    cut_count = max(0, len(scenes) - 1)

    # "Cuts per second" — the primary editing-pace metric.
    # Higher values = faster editing.  A typical TikTok might have
    # 0.3–1.5 cuts/sec; a single unbroken vlog shot has 0.
    cuts_per_second = cut_count / duration if duration > 0 else 0.0

    # Scene durations (in seconds).
    scene_durations = [end - start for start, end in scenes]

    if scene_durations:
        avg_scene_duration = sum(scene_durations) / len(scene_durations)
        min_scene_duration = min(scene_durations)
        max_scene_duration = max(scene_durations)
    else:
        # No scenes detected (e.g. video too short or unreadable).
        avg_scene_duration = duration
        min_scene_duration = duration
        max_scene_duration = duration

    return {
        "cut_count": cut_count,
        "cuts_per_second": round(cuts_per_second, 4),
        "avg_scene_duration": round(avg_scene_duration, 4),
        "min_scene_duration": round(min_scene_duration, 4),
        "max_scene_duration": round(max_scene_duration, 4),
        # Store raw timestamps as JSON so they can be loaded later for
        # frame-level analysis or visualization.
        "scene_timestamps": json.dumps(
            [(round(s, 3), round(e, 3)) for s, e in scenes]
        ),
    }


def extract(video_path: Path, video_id: str) -> Dict:
    """
    Full cut-detection pipeline for a single video.

    This is the public API that the orchestrator (extract_all.py) calls.
    It downloads or receives a local video path, runs scene detection,
    and returns a flat dictionary of features ready to be written to CSV.

    Args:
        video_path: Path to a local video file.
        video_id:   Unique video identifier (for the output CSV key column).

    Returns:
        Dictionary with keys: video_id, cut_count, cuts_per_second,
        avg_scene_duration, min_scene_duration, max_scene_duration,
        scene_timestamps, duration_sec, fps.
    """
    # Run the detector.
    scenes, duration, fps = detect_scenes(video_path)

    # Compute summary features.
    features = compute_cut_features(scenes, duration)

    # Add metadata columns.
    features["video_id"] = video_id
    features["duration_sec"] = round(duration, 4)
    features["fps"] = round(fps, 2)

    return features
