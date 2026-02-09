# Feature Extraction Pipeline

Extracts computer-vision features from TikTok videos stored in GCS.
Each extractor produces a CSV keyed on `video_id` for joining with annotation data.

## Quick Start

```bash
cd feature_extraction
pip install -r requirements.txt

# Run cut detection on first 10 videos
python extract_all.py --extractors cuts --limit 10

# Run all extractors on every video
python extract_all.py
```

## Extractors

| Name | File | Status | Output columns |
|------|------|--------|----------------|
| Cut detection | `extractors/cut_detection.py` | Working | `cut_count`, `cuts_per_second`, `avg_scene_duration`, `min_scene_duration`, `max_scene_duration`, `scene_timestamps` |
| Object density | `extractors/density.py` | Stub | `num_objects`, `color_entropy`, `edge_density_ratio`, `avg_motion_magnitude` |
| Gaze | `extractors/gaze.py` | Stub | `num_faces`, `gaze_at_camera_ratio`, `avg_gaze_pitch`, `avg_gaze_yaw` |
| Object detection | `extractors/object_detection.py` | Stub | `num_objects`, `object_types`, `num_humans` |
| Text detection | `extractors/text_detection.py` | Stub | `has_text`, `text_area_ratio`, `text_changes_per_second` |

## Output

Feature CSVs are written to `data/features/{extractor_name}.csv`.

## Architecture

```
extract_all.py          # Orchestrator — downloads videos, runs extractors, writes CSVs
config.py               # Pipeline settings (thresholds, paths)
extractors/
  cut_detection.py      # PySceneDetect-based scene change detection
  density.py            # Multi-dimensional visual density (stub)
  gaze.py               # Face + gaze estimation (stub)
  object_detection.py   # YOLOv8 object detection (stub)
  text_detection.py     # EasyOCR text detection (stub)
```
