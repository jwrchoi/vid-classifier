"""
Gaze Estimation Extractor
===========================

Detects faces and estimates gaze direction using MediaPipe Face Mesh,
then measures how often subjects look directly at the camera.

WHY THIS MATTERS FOR RESEARCH
------------------------------
Direct gaze at the camera is a key indicator of "2nd person" perspective
and parasocial interaction.  When a TikTok creator looks directly into the
lens, viewers perceive it as eye contact — the most powerful parasocial cue.
This metric quantifies that behavior across an entire video.

HOW IT WORKS (COMPUTER VISION CONCEPTS)
-----------------------------------------

**MediaPipe Face Mesh** detects 478 3D face landmarks in real time:
  - 468 standard face landmarks (eyebrows, nose, lips, jaw, cheeks)
  - 10 iris landmarks (5 per eye) added in the "refine_landmarks" mode

We use the **iris landmarks** to estimate gaze direction:

1. **Iris center** (landmark 468 for left eye, 473 for right eye):
   The 2D position of the iris center in the image.

2. **Eye corners** (landmarks 33/133 for left, 362/263 for right):
   Define the horizontal extent of the eye opening.

3. **Gaze ratio**: How far the iris is from the inner corner vs. the
   full eye width.  A ratio near 0.5 means the iris is centered (looking
   straight ahead / at camera).  Closer to 0 or 1 means looking left or right.

4. **Pitch estimation**: We compare the iris vertical position to the
   eye's vertical midpoint.  Iris above center → looking up, below → down.

5. **"At camera" classification**: If both the horizontal gaze ratio and
   vertical offset are within a tolerance of center, we classify the gaze
   as directed at the camera.

This iris-based approach avoids the need for a separate gaze model (like
L2CS-Net) and works reliably on the kinds of front-facing shots common
in TikTok videos.

OUTPUT COLUMNS
--------------
- video_id:             Unique video identifier
- num_faces:            Average number of faces detected per frame
- gaze_at_camera_ratio: Fraction of face-frames where gaze is directed at camera
- avg_gaze_yaw:         Average horizontal gaze deviation (degrees, 0 = center)
- avg_gaze_pitch:       Average vertical gaze deviation (degrees, 0 = center)
- num_frames_sampled:   How many frames were analyzed

DEPENDENCIES
------------
- mediapipe      pip install mediapipe
- opencv-python
- numpy

NOTE: Requires the face_landmarker.task model file.  The extractor looks for
it at feature_extraction/face_landmarker.task (downloaded automatically from
Google's model hub if missing).
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from feature_extraction.config import (
    FRAME_SAMPLE_INTERVAL,
    MAX_FRAMES_PER_VIDEO,
    MEDIAPIPE_MAX_FACES,
    MEDIAPIPE_FACE_CONFIDENCE,
    GAZE_AT_CAMERA_TOLERANCE_DEG,
)

# Path to the MediaPipe FaceLandmarker model file (.task format).
_MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Lazy-load MediaPipe FaceLandmarker (new tasks API, mediapipe >= 0.10.8).
# ---------------------------------------------------------------------------
_face_landmarker = None


def _get_face_landmarker():
    """Load MediaPipe FaceLandmarker (cached after first call)."""
    global _face_landmarker
    if _face_landmarker is None:
        import mediapipe as mp

        # Download model if not present locally.
        if not _MODEL_PATH.exists():
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            print(f"Downloading face_landmarker model to {_MODEL_PATH} ...")
            urllib.request.urlretrieve(url, str(_MODEL_PATH))

        base_options = mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=MEDIAPIPE_MAX_FACES,
            min_face_detection_confidence=MEDIAPIPE_FACE_CONFIDENCE,
            min_face_presence_confidence=MEDIAPIPE_FACE_CONFIDENCE,
        )
        _face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def estimate_gaze_for_face(
    landmarks,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, bool]:
    """
    Estimate gaze direction for a single detected face using iris landmarks.

    The idea:
    - Each eye has an iris center landmark and two corner landmarks.
    - We compute where the iris sits relative to the eye opening.
    - If the iris is near the center both horizontally and vertically,
      the person is looking at the camera.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList for one face.
        img_w:     Image width in pixels.
        img_h:     Image height in pixels.

    Returns:
        Tuple of (yaw_degrees, pitch_degrees, is_at_camera).
        yaw_degrees:  Horizontal deviation from center (0 = straight ahead).
                      Positive = looking right, negative = looking left.
        pitch_degrees: Vertical deviation from center (0 = straight ahead).
                      Positive = looking down, negative = looking up.
        is_at_camera: True if gaze is within the tolerance of center.
    """
    # Helper to convert normalized landmark to pixel coordinates.
    def lm_px(idx):
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    # -----------------------------------------------------------------------
    # Iris and eye corner landmarks (MediaPipe Face Mesh indices).
    #
    # Left eye (from the subject's perspective):
    #   Iris center: 468,  Inner corner: 133,  Outer corner: 33
    # Right eye:
    #   Iris center: 473,  Inner corner: 362,  Outer corner: 263
    #
    # "Inner" = closer to the nose, "Outer" = closer to the temple.
    # -----------------------------------------------------------------------
    left_iris = lm_px(468)
    left_inner = lm_px(133)
    left_outer = lm_px(33)

    right_iris = lm_px(473)
    right_inner = lm_px(362)
    right_outer = lm_px(263)

    # --- Horizontal gaze (yaw) ---
    # Compute the ratio: how far the iris is from the inner corner
    # relative to the full eye width.  0.5 = centered.
    def horiz_ratio(iris, inner, outer):
        eye_width = math.dist(inner, outer)
        if eye_width < 1:
            return 0.5  # degenerate case
        iris_offset = math.dist(inner, iris)
        return iris_offset / eye_width

    left_ratio = horiz_ratio(left_iris, left_inner, left_outer)
    right_ratio = horiz_ratio(right_iris, right_inner, right_outer)
    avg_ratio = (left_ratio + right_ratio) / 2

    # Convert ratio to approximate yaw angle.
    # Ratio 0.5 → 0 degrees.  Ratio 0.0 or 1.0 → ~45 degrees.
    yaw_deg = (avg_ratio - 0.5) * 90  # linear approximation

    # --- Vertical gaze (pitch) ---
    # Compare iris Y to eye vertical midpoint.
    # Use upper/lower eyelid landmarks to define the vertical extent.
    # Left eye:  top=159, bottom=145
    # Right eye: top=386, bottom=374
    def vert_ratio(iris, top_lm_idx, bottom_lm_idx):
        top_y = lm_px(top_lm_idx)[1]
        bot_y = lm_px(bottom_lm_idx)[1]
        eye_height = bot_y - top_y
        if eye_height < 1:
            return 0.5
        return (iris[1] - top_y) / eye_height

    left_vert = vert_ratio(left_iris, 159, 145)
    right_vert = vert_ratio(right_iris, 386, 374)
    avg_vert = (left_vert + right_vert) / 2

    pitch_deg = (avg_vert - 0.5) * 90  # positive = looking down

    # --- "At camera" classification ---
    is_at_camera = (
        abs(yaw_deg) <= GAZE_AT_CAMERA_TOLERANCE_DEG
        and abs(pitch_deg) <= GAZE_AT_CAMERA_TOLERANCE_DEG
    )

    return (round(yaw_deg, 2), round(pitch_deg, 2), is_at_camera)


def analyze_frame(frame_bgr: np.ndarray) -> Dict:
    """
    Detect faces and estimate gaze for each face in a single frame.

    Args:
        frame_bgr: Single video frame in BGR format.

    Returns:
        Dict with keys:
        - num_faces:    Number of detected faces.
        - gazes:        List of (yaw, pitch, is_at_camera) per face.
    """
    import mediapipe as mp

    landmarker = _get_face_landmarker()

    # MediaPipe tasks API expects an mp.Image object in RGB format.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    img_h, img_w = frame_bgr.shape[:2]

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"num_faces": 0, "gazes": []}

    gazes = []
    for face_landmarks in result.face_landmarks:
        # face_landmarks is a list of NormalizedLandmark objects.
        yaw, pitch, at_camera = estimate_gaze_for_face(face_landmarks, img_w, img_h)
        gazes.append((yaw, pitch, at_camera))

    return {
        "num_faces": len(gazes),
        "gazes": gazes,
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
    Full gaze-estimation pipeline for a single video.

    Samples frames, detects faces, estimates gaze direction, and
    computes the "gaze at camera" ratio.

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
            "num_faces": None,
            "gaze_at_camera_ratio": None,
            "avg_gaze_yaw": None,
            "avg_gaze_pitch": None,
            "num_frames_sampled": 0,
        }

    all_face_counts = []
    all_yaws = []
    all_pitches = []
    at_camera_count = 0
    total_face_observations = 0

    for frame in frames:
        result = analyze_frame(frame)
        all_face_counts.append(result["num_faces"])

        for yaw, pitch, at_camera in result["gazes"]:
            all_yaws.append(yaw)
            all_pitches.append(pitch)
            total_face_observations += 1
            if at_camera:
                at_camera_count += 1

    # Gaze-at-camera ratio: what fraction of all face-observations
    # had the subject looking at the camera?
    gaze_ratio = (
        at_camera_count / total_face_observations
        if total_face_observations > 0
        else 0.0
    )

    return {
        "video_id": video_id,
        "num_faces": round(float(np.mean(all_face_counts)), 2),
        "gaze_at_camera_ratio": round(gaze_ratio, 4),
        "avg_gaze_yaw": round(float(np.mean(all_yaws)), 2) if all_yaws else 0.0,
        "avg_gaze_pitch": round(float(np.mean(all_pitches)), 2) if all_pitches else 0.0,
        "num_frames_sampled": len(frames),
    }
