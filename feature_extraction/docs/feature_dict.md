# Feature Dictionary

Reference for all features produced by the extraction pipeline.
Each extractor writes a separate CSV to `data/features/<extractor_name>.csv`, keyed on `video_id`.

---

## Global Sampling Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FRAME_SAMPLE_INTERVAL` | 15 | Sample every 15th frame (~2 fps at 30 fps video) |
| `MAX_FRAMES_PER_VIDEO` | 200 | Cap to prevent OOM on long videos |

All frame-level extractors (density, object detection, text detection, gaze) use these shared sampling parameters. Cut detection reads every frame (required for consecutive-frame comparison).

---

## 1. Cut Detection (`cuts.csv`)

Detects hard scene transitions using PySceneDetect's ContentDetector.

**Method:** Compares consecutive frames in HSV color space. A cut is declared when the per-channel mean pixel intensity difference exceeds a threshold.

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `cut_count` | int | Number of detected scene changes. Equals `len(scenes) - 1`. |
| `cuts_per_second` | float | Editing pace: `cut_count / duration_sec` |
| `avg_scene_duration` | float | Mean scene length in seconds: `duration_sec / len(scenes)` |
| `min_scene_duration` | float | Shortest scene (seconds) |
| `max_scene_duration` | float | Longest scene (seconds) |
| `scene_timestamps` | JSON str | List of `[start_sec, end_sec]` tuples per scene |
| `duration_sec` | float | Total video duration (seconds) |
| `fps` | float | Video frame rate |

**Key parameters:**
- `CONTENT_DETECTOR_THRESHOLD = 27.0` — higher = fewer cuts detected
- `MIN_SCENE_LENGTH_FRAMES = 10` — scenes shorter than this are merged (filters flashes/glitches)

---

## 2. Visual Density (`density.csv`)

Measures visual complexity across three dimensions.

### 2a. Color Entropy

**Method:** Convert frame to HSV. Compute a normalized histogram (64 bins) per channel. Apply Shannon entropy.

$$H = -\sum_{i=1}^{B} p_i \cdot \log_2(p_i)$$

where $p_i$ is the probability in bin $i$ and $B$ = `COLOR_HISTOGRAM_BINS` (64).

The reported value is the mean of the H, S, and V channel entropies, then averaged across all sampled frames.

- **High entropy** (~5-6 bits): many colors spread evenly (complex scenes, colorful overlays)
- **Low entropy** (~2-3 bits): few dominant colors (solid backgrounds, simple scenes)

### 2b. Edge Density

**Method:** Convert frame to grayscale. Apply Canny edge detector. Count edge pixels.

$$\text{edge\_density\_ratio} = \frac{\text{edge pixels}}{\text{total pixels}}$$

Canny uses hysteresis thresholding with `CANNY_LOW = 100` and `CANNY_HIGH = 200`:
- Gradient > 200: definitely an edge
- Gradient 100-200: edge only if connected to a strong edge
- Gradient < 100: not an edge

- **High ratio** (~0.08-0.15): lots of detail, text overlays, textures
- **Low ratio** (~0.01-0.03): smooth surfaces, solid colors, blurred backgrounds

### 2c. Motion Magnitude

**Method:** Farneback dense optical flow between consecutive sampled frames. Computes a 2D displacement vector $(dx, dy)$ at every pixel.

$$\text{magnitude} = \sqrt{dx^2 + dy^2}$$

The reported value is the mean magnitude across all pixels, averaged across all consecutive frame pairs.

- **High magnitude** (~10-30 px): fast camera movement, zooms, quick subject motion
- **Low magnitude** (~0-3 px): static camera, minimal movement

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `color_entropy` | float | Mean Shannon entropy of HSV histograms (bits) |
| `edge_density_ratio` | float | Mean fraction of edge pixels per frame (0 to 1) |
| `avg_motion_magnitude` | float | Mean optical flow magnitude (pixels of displacement) |
| `num_frames_sampled` | int | Number of frames analyzed |

---

## 3. Object Detection (`object_detection.csv`)

Detects and counts objects using YOLOv8-nano, pre-trained on COCO (80 classes).

**Method:** Run YOLOv8n inference on each sampled frame. Filter detections by confidence threshold. Aggregate counts across frames.

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `avg_objects` | float | Mean number of detected objects per frame |
| `num_humans` | float | Mean number of "person" class detections per frame |
| `object_types` | JSON str | Sorted list of unique COCO class names seen across all frames |
| `num_frames_sampled` | int | Number of frames analyzed |

**Key parameters:**
- `YOLO_MODEL_NAME = "yolov8n.pt"` — nano model (~6 MB, fastest)
- `YOLO_CONFIDENCE_THRESHOLD = 0.25` — minimum detection confidence

**Notes:**
- `num_humans` is especially relevant for parasocial interaction research (single face vs. group shots)
- `object_types` is a JSON array (e.g., `["bottle", "cell phone", "person"]`) for downstream categorical analysis

---

## 4. Text Detection (`text_detection.csv`)

Detects and reads on-screen text using EasyOCR (CRAFT detection + CRNN recognition).

### 4a. Text Area Ratio

$$\text{text\_area\_ratio} = \frac{1}{N}\sum_{f=1}^{N} \frac{\sum \text{bbox area}_f}{\text{frame area}_f}$$

Bounding box area is computed from the axis-aligned bounding rectangle of each detected text polygon.

### 4b. Text Changes Per Second

Measures how often on-screen text content changes between consecutive sampled frames.

$$\text{text\_changes\_per\_second} = \frac{\text{num\_changes}}{(N-1) \times (\text{FRAME\_SAMPLE\_INTERVAL} / \text{fps})}$$

A "change" occurs when the set of recognized text strings (lowercased) differs between two consecutive sampled frames.

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `has_text` | bool | Whether any text was detected in any frame |
| `text_area_ratio` | float | Mean fraction of frame area covered by text bounding boxes |
| `avg_text_regions` | float | Mean number of text regions detected per frame |
| `text_changes_per_second` | float | Rate of text content changes (higher = faster text transitions) |
| `num_frames_sampled` | int | Number of frames analyzed |

**Key parameters:**
- `EASYOCR_LANGUAGES = ["en"]` — English text only
- `EASYOCR_CONFIDENCE_THRESHOLD = 0.3` — minimum OCR confidence

---

## 5. Gaze Estimation (`gaze.csv`)

Detects faces and estimates gaze direction using MediaPipe FaceLandmarker with iris tracking.

### 5a. Gaze Direction

Uses iris center landmarks (468 for left eye, 473 for right eye) relative to eye corner landmarks to estimate where the subject is looking.

**Horizontal (yaw):**

$$\text{gaze\_ratio} = \frac{\text{dist}(\text{inner corner}, \text{iris center})}{\text{dist}(\text{inner corner}, \text{outer corner})}$$

$$\text{yaw\_deg} = (\text{gaze\_ratio} - 0.5) \times 90°$$

A ratio of 0.5 (iris centered) = 0 degrees (looking straight ahead).

**Vertical (pitch):**

$$\text{vert\_ratio} = \frac{\text{iris}_y - \text{top eyelid}_y}{\text{bottom eyelid}_y - \text{top eyelid}_y}$$

$$\text{pitch\_deg} = (\text{vert\_ratio} - 0.5) \times 90°$$

Positive pitch = looking down; negative = looking up.

### 5b. "At Camera" Classification

A face-frame is classified as "looking at camera" when:

$$|\text{yaw\_deg}| \leq T \quad \text{AND} \quad |\text{pitch\_deg}| \leq T$$

where $T$ = `GAZE_AT_CAMERA_TOLERANCE_DEG` (15.0 degrees).

$$\text{gaze\_at\_camera\_ratio} = \frac{\text{face-frames classified as at-camera}}{\text{total face-frames}}$$

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `num_faces` | float | Mean number of faces detected per frame |
| `gaze_at_camera_ratio` | float | Fraction of face-frames with gaze directed at camera (0 to 1) |
| `avg_gaze_yaw` | float | Mean horizontal gaze deviation (degrees, 0 = center) |
| `avg_gaze_pitch` | float | Mean vertical gaze deviation (degrees, 0 = center) |
| `num_frames_sampled` | int | Number of frames analyzed |

**Key parameters:**
- `MEDIAPIPE_MAX_FACES = 4` — max faces detected per frame
- `MEDIAPIPE_FACE_CONFIDENCE = 0.5` — minimum detection confidence
- `GAZE_AT_CAMERA_TOLERANCE_DEG = 15.0` — degrees from center to count as "at camera"

**Research relevance:** `gaze_at_camera_ratio` is the primary metric for "2nd person perspective" and parasocial interaction — direct gaze simulates eye contact with the viewer.

---

## Joining Features with Annotations

All CSVs share the `video_id` column, enabling straightforward joins:

```python
import pandas as pd

annotations = pd.read_csv("data/annotations.csv")
cuts = pd.read_csv("data/features/cuts.csv")
density = pd.read_csv("data/features/density.csv")

merged = annotations.merge(cuts, on="video_id").merge(density, on="video_id")
```
