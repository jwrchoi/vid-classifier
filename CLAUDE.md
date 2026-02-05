# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**Running Shoe Video Classifier** - A Streamlit-based annotation tool for coding TikTok videos about running shoes. Used for parasocial interaction research.

### Purpose
Human coders annotate videos for:
- **Perspective (POV)**: 1st person, 2nd person, 3rd person, NA
- **Social Distance**: Personal, Social, Public, NA

Model predictions exist but are **NOT shown to coders** to avoid bias.

## Key Paths

| What | Path |
|------|------|
| Sample Videos | `/Volumes/T9/running_brand_videos/01_media/01_sample-videos/` |
| Trained Models | `/Users/roycechoi/proj/tiktok_analysis/codes/artifacts/model_exports/resnet50_models/` |
| Annotations Output | `./data/annotations.csv` |

## Models (v7)

- `pov_resnet_50_v7.pth` - Predicts: 1st person, 2nd person, 3rd person
- `social_distance_resnet_50_v7.pth` - Predicts: Personal, Social, Public

## Architecture

```
running_shoe_vid_classifier/
├── app.py                  # Main Streamlit dashboard
├── config.py               # Configuration (paths, models, device)
├── CLAUDE.md               # This file
├── models/
│   ├── model_loader.py     # ResNet-50 loading & inference
│   └── feature_extractors.py  # (Not used in current UI)
├── utils/
│   ├── video_processing.py # Frame sampling
│   └── database.py         # CSV annotation storage
├── batch_preprocess.py     # Pre-compute predictions (optional)
├── requirements.txt        # Dependencies
└── data/                   # Created at runtime
    └── annotations.csv     # Human annotations output
```

## Running the App

```bash
# Activate virtual environment
source venv/bin/activate

# Run dashboard
streamlit run app.py
```

## Key Design Decisions

1. **No model predictions shown to coders** - Prevents anchoring bias
2. **"No human visible" checkbox** - Screener question; affects which annotation fields are applicable
3. **Multi-category options** - Not binary; includes NA for edge cases
4. **CSV output** - Simple, easy to inspect, compatible with pandas

## Annotation Schema

| Field | Options |
|-------|---------|
| `no_human_visible` | Boolean checkbox |
| `perspective` | "1st person", "2nd person", "3rd person", "NA" |
| `distance` | "Personal", "Social", "Public", "NA" |
| `notes` | Free text |
| `is_difficult` | Boolean flag |

## Common Tasks

### Add a new annotator
Just enter name in sidebar - no setup needed.

### Export annotations
Annotations auto-save to `data/annotations.csv`.

### Pre-process videos (optional, for speed)
```bash
python batch_preprocess.py --limit 50
```

## Dependencies

- Python 3.11+
- PyTorch (with MPS support for Apple Silicon)
- Streamlit
- OpenCV
- pandas, numpy

## Contact

Project by Royce Choi for TikTok parasocial interaction research.
