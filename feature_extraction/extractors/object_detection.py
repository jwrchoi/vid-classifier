"""
Object Detection Extractor (STUB)
===================================

Will use YOLOv8 to count and classify objects in sampled frames.

Planned approach:
1. Sample frames at a configurable interval (e.g. every 15 frames).
2. Run YOLOv8 (nano or small variant) on each frame.
3. Aggregate detections across frames:
   - num_objects:  Average number of detected objects per frame.
   - object_types: Set of unique COCO class names seen across all frames.
   - num_humans:   Average number of "person" detections per frame.

YOLOv8 uses the COCO dataset's 80 object classes (person, bicycle, car,
dog, cell phone, etc.).  The "person" class is especially useful for
parasocial interaction research.

Dependencies (install when implementing):
    pip install ultralytics
"""

from pathlib import Path
from typing import Dict


def extract(video_path: Path, video_id: str) -> Dict:
    """Extract object detection features from a video. NOT YET IMPLEMENTED."""
    raise NotImplementedError(
        "object_detection extractor is not yet implemented. "
        "See the docstring for planned features."
    )
