# TikTok Video Analysis

Monorepo for parasocial interaction research on TikTok running-shoe content.

## Sub-projects

| Directory | Description | Status |
|-----------|-------------|--------|
| `annotation_dashboard/` | Streamlit tool for human coders to annotate videos (POV, social distance) | Production (Cloud Run) |
| `feature_extraction/` | CV pipeline to extract cut pace, density, gaze, objects, text from videos | Cut detection working; others stubbed |
| `model_training/` | Fine-tune ResNet-50 classifiers on human annotations | Placeholder |
| `shared/` | GCS utilities, video processing, and config shared across sub-projects | Working |

## Quick Start

```bash
# Annotation dashboard
cd annotation_dashboard
pip install -r requirements.txt
streamlit run app.py

# Feature extraction (cut detection example)
cd feature_extraction
pip install -r requirements.txt
python extract_all.py --extractors cuts --limit 10
```

## Data

All data lives in the `data/` directory (GCS FUSE-mounted on Cloud Run):
- `annotations.csv` — Human annotations
- `video_list_v2.csv` — Fixed list of videos to annotate
- `features/` — Extracted feature CSVs (one per extractor, keyed on `video_id`)

Videos are stored in GCS bucket `gs://vid-classifier-db/videos/`.

## Architecture

See [CLAUDE.md](CLAUDE.md) for the full project guide.
