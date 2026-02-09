# Running Shoe Video Classifier

An interactive annotation tool for classifying TikTok videos about running shoes based on visual and stylistic features.

## Features

- **Video Playback**: Watch videos directly in the browser
- **Model Predictions**: Pre-trained ResNet-50 models predict perspective, distance, and gaze
- **Computed Features**: Automatic detection of editing pace, visual density, and gestures
- **Annotation Interface**: Easy form-based annotation with model suggestions
- **Progress Tracking**: See how many videos have been annotated
- **Export**: Export annotations for model retraining

## Quick Start

### 1. Install Dependencies

```bash
cd running_shoe_classifier
pip install -r requirements.txt
```

**Note for Mac users with Apple Silicon**: PyTorch will automatically use MPS (Metal Performance Shaders) for acceleration.

### 2. Configure Paths

Edit `config.py` and update these paths:

```python
# Directory containing your video files
VIDEO_DIR = Path("/path/to/your/videos")

# Directory containing your trained model weights (.pth files)
MODELS_DIR = Path("/path/to/your/models")
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 4. (Recommended) Pre-process Videos for Instant Loading

For a smooth demo experience, pre-process videos first:

```bash
# Process all videos in VIDEO_DIR
python batch_preprocess.py

# Or process just 20 videos for a quick demo
python batch_preprocess.py --limit 20

# Or specify a custom directory with sample videos
python batch_preprocess.py /path/to/demo/videos --limit 20
```

This creates `data/predictions_cache.csv`. The app will load predictions instantly!

**Time estimates:**
- 10 videos: ~1-2 minutes
- 20 videos: ~2-4 minutes
- 50 videos: ~5-10 minutes

### 5. Share with Colleagues

To share the app on your local network:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then share your IP address with colleagues: `http://YOUR_IP:8501`

To find your IP:
- Mac: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- Windows: `ipconfig`

## Directory Structure

```
running_shoe_classifier/
├── app.py                      # Main Streamlit application
├── batch_preprocess.py         # Pre-compute predictions (run before demo)
├── setup.py                    # Verify installation
├── config.py                   # Configuration (paths, models, etc.)
├── requirements.txt            # Python dependencies
├── coding_instructions.md      # Instructions for annotators
├── models/
│   ├── __init__.py
│   ├── model_loader.py         # Load trained ResNet models
│   └── feature_extractors.py   # Editing pace, density, gestures
├── utils/
│   ├── __init__.py
│   ├── video_processing.py     # Frame sampling, video loading
│   └── database.py             # Annotation storage
└── data/                       # Created automatically
    ├── annotations.csv         # Saved annotations
    └── predictions_cache.csv   # Pre-computed predictions
```

## Expected Model Files

Place your trained models in the `MODELS_DIR` directory:

- `pov_resnet_50_binary_v1.pth` - Perspective (binary)
- `social_distance_resnet_50_binary_v1.pth` - Distance (binary)
- `gaze_resnet_50_v7.pth` - Gaze direction

## Features Being Annotated

| Feature | Options | Description |
|---------|---------|-------------|
| Perspective | 1st, 2nd, 3rd person | Camera POV |
| Distance | Close, Mid/Wide | Camera proximity to subject |
| Gaze | At camera, Away, No face | Where subject is looking |
| Editing Pace | Slow, Moderate, Fast | Cut frequency |
| Visual Density | Minimal, Moderate, High | Visual complexity |
| Gesture | Hands visible, Not visible, Pointing | Hand presence |

## Troubleshooting

### "Models directory not found"
Update `MODELS_DIR` in `config.py` to point to your model files.

### "No videos found"
Update `VIDEO_DIR` in `config.py` to point to your video folder.

### MediaPipe warnings
MediaPipe is optional. The app works without it (gesture/face detection will be disabled).

### Slow video loading
The app samples frames for analysis. Adjust `FRAME_SAMPLE_INTERVAL` in `config.py` to sample fewer frames.

## Exporting Annotations

Annotations are saved to `data/annotations.csv` automatically. You can also:

1. Click "Export Annotations" in the sidebar
2. Use the database module directly:

```python
from utils.database import AnnotationDatabase
db = AnnotationDatabase("data")
df = db.export_for_training("training_data.csv")
```

## Next Steps

After collecting annotations:

1. Export annotations for model retraining
2. Identify disagreements between model and human labels
3. Fine-tune models on corrected annotations
4. Repeat the annotation-training cycle

---

Built for parasocial interaction research on TikTok content.
