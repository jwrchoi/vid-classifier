"""
Gaze Estimation Extractor (STUB)
==================================

Will detect faces and estimate gaze direction to measure how often
the subject looks directly at the camera ("gaze at camera" ratio).

Planned approach:
1. Face detection via MediaPipe Face Mesh (fast, works on CPU).
2. Gaze estimation via L2CS-Net (pre-trained model that predicts
   pitch and yaw angles of gaze direction from a face crop).
3. A face is considered "looking at camera" if both pitch and yaw
   are within a configurable threshold of (0, 0).

Output columns:
    num_faces, gaze_at_camera_ratio, avg_gaze_pitch, avg_gaze_yaw

Dependencies (install when implementing):
    pip install mediapipe l2cs-net
"""

from pathlib import Path
from typing import Dict


def extract(video_path: Path, video_id: str) -> Dict:
    """Extract gaze features from a video. NOT YET IMPLEMENTED."""
    raise NotImplementedError(
        "gaze extractor is not yet implemented. "
        "See the docstring for planned features."
    )
