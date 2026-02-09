# Feature extractors package.
#
# Each module exposes an `extract(video_path, video_id) -> dict` function
# that returns a flat dictionary of features for one video.

from .cut_detection import extract as extract_cuts
