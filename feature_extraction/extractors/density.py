"""
Visual Density Extractor (STUB)
================================

Will compute multiple dimensions of visual "density" / complexity:

- Color complexity:  Histogram entropy in HSV space.
  High entropy = many colors = visually complex frame.
- Edge density:      Ratio of edge pixels (via Canny edge detector) to total pixels.
  High ratio = lots of detail / texture.
- Motion magnitude:  Average optical flow magnitude between consecutive frames
  (Farneback dense optical flow).  High motion = fast movement or camera shake.

Dependencies (install when implementing):
    pip install scipy opencv-python numpy
"""

from pathlib import Path
from typing import Dict


def extract(video_path: Path, video_id: str) -> Dict:
    """Extract visual density features from a video. NOT YET IMPLEMENTED."""
    raise NotImplementedError(
        "density extractor is not yet implemented. "
        "See the docstring for planned features."
    )
