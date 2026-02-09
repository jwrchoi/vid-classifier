# Feature extractors package.
#
# Each module exposes an `extract(video_path, video_id) -> dict` function
# that returns a flat dictionary of features for one video.

from .cut_detection import extract as extract_cuts
from .density import extract as extract_density
from .object_detection import extract as extract_objects
from .text_detection import extract as extract_text
from .gaze import extract as extract_gaze
