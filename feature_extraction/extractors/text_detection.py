"""
Text Detection Extractor (STUB)
=================================

Will use EasyOCR to detect and measure on-screen text in video frames.

Planned approach:
1. Sample frames at a configurable interval.
2. Run EasyOCR on each frame to get bounding boxes + recognized text.
3. Compute per-frame metrics:
   - has_text:              Boolean — any text detected?
   - text_area_ratio:       Total area of text bounding boxes / frame area.
   - text_position:         Dominant position of text (top, center, bottom).
   - avg_text_height:       Average height of text boxes in pixels.
4. Compute cross-frame metrics:
   - text_changes_per_second: How often the text content changes between
     consecutive sampled frames.  High values indicate fast-changing captions
     or subtitle overlays common in TikTok product reviews.

Dependencies (install when implementing):
    pip install easyocr
"""

from pathlib import Path
from typing import Dict


def extract(video_path: Path, video_id: str) -> Dict:
    """Extract text detection features from a video. NOT YET IMPLEMENTED."""
    raise NotImplementedError(
        "text_detection extractor is not yet implemented. "
        "See the docstring for planned features."
    )
