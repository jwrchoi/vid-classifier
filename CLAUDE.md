# CLAUDE.md - Project Guide for Claude Code

## Project Overview

**TikTok Video Analysis** - A monorepo for parasocial interaction research on TikTok running-shoe content. Contains an annotation dashboard, a feature extraction pipeline, and model training code.

### Purpose
Human coders annotate videos for:
- **Perspective (POV)**: 1st person, 2nd person, 3rd person, NA
- **Social Distance**: Personal, Social, Public, NA

Automated feature extraction captures cut pace, visual density, gaze direction, and more.

Model predictions exist but are **NOT shown to coders** to avoid bias.

## Key Paths

| What | Path |
|------|------|
| Videos | GCS bucket `vid-classifier-db` (prefix `videos/`) |
| Trained Models | `/Users/roycechoi/proj/tiktok_analysis/codes/artifacts/model_exports/resnet50_models/` |
| Annotations Output | `./data/annotations.csv` (GCS FUSE-mounted on Cloud Run) |
| Feature CSVs | `./data/features/*.csv` |
| Video List | `./data/video_list_v2.csv` |

## Monorepo Architecture

```
tiktok_video_analysis/
├── annotation_dashboard/        # Streamlit annotation tool (deployed to Cloud Run)
│   ├── app.py                   # Main Streamlit dashboard
│   ├── config.py                # Dashboard-specific config (models, device, UI)
│   ├── Dockerfile               # Container image for Cloud Run
│   ├── deploy.sh                # End-to-end GCP deployment script
│   ├── requirements.txt         # Dashboard dependencies
│   ├── models/
│   │   ├── model_loader.py      # ResNet-50 loading & inference
│   │   └── feature_extractors.py
│   ├── utils/
│   │   ├── video_processing.py  # Frame sampling
│   │   ├── gcs.py               # GCS video streaming (Streamlit-cached)
│   │   └── database.py          # CSV annotation storage (per-coder)
│   ├── scripts/
│   │   └── build_filtered_video_list.py
│   ├── batch_preprocess.py      # Pre-compute predictions (optional)
│   └── docs/                    # Session logs
│
├── feature_extraction/          # CV feature extraction pipeline
│   ├── extract_all.py           # Orchestrator (downloads from GCS, runs extractors)
│   ├── config.py                # Pipeline settings (thresholds, paths)
│   ├── requirements.txt
│   ├── README.md
│   └── extractors/
│       ├── cut_detection.py     # PySceneDetect scene-change detection (working)
│       ├── density.py           # Visual density: color entropy, edges, motion (stub)
│       ├── gaze.py              # Face + gaze estimation via MediaPipe/L2CS (stub)
│       ├── object_detection.py  # YOLOv8 object counting (stub)
│       └── text_detection.py    # EasyOCR text detection (stub)
│
├── model_training/              # Model fine-tuning (placeholder)
│   ├── fine_tune_pov.py
│   ├── fine_tune_distance.py
│   └── requirements.txt
│
├── shared/                      # Shared utilities used across sub-projects
│   ├── __init__.py
│   ├── config.py                # GCS bucket, data paths, video extensions
│   ├── gcs_utils.py             # GCS client, blob listing, download/upload
│   └── video_utils.py           # Video metadata, frame sampling, ID extraction
│
├── data/                        # Created at runtime (GCS FUSE on Cloud Run)
│   ├── annotations.csv          # Human annotations output
│   ├── video_list_v2.csv        # Fixed video list
│   └── features/                # Extracted feature CSVs
│       └── cuts.csv             # Cut detection output
│
├── CLAUDE.md                    # This file
└── .gitignore
```

## Running the App

```bash
# Annotation dashboard (local development)
cd annotation_dashboard
pip install -r requirements.txt
streamlit run app.py

# Deploy dashboard to Cloud Run
cd annotation_dashboard
bash deploy.sh

# Feature extraction
cd feature_extraction
pip install -r requirements.txt
python extract_all.py --extractors cuts --limit 10
```

## Models (v7)

- `pov_resnet_50_v7.pth` - Predicts: 1st person, 2nd person, 3rd person
- `social_distance_resnet_50_v7.pth` - Predicts: Personal, Social, Public

Models are optional — the dashboard runs without PyTorch installed (torch/ModelLoader imports are wrapped in try/except).

## Deployment (Cloud Run)

`annotation_dashboard/deploy.sh` handles everything end-to-end:
1. Enables GCP APIs (Cloud Run, Cloud Build, Artifact Registry, Storage)
2. Creates Artifact Registry repo and service account
3. Uploads seed data to GCS
4. Builds Docker image via Cloud Build
5. Deploys to Cloud Run with GCS FUSE volume mount

The GCS bucket `vid-classifier-db` is mounted at `/app/data`, so `annotations.csv` is read/written directly to GCS with no separate sync step.

## Key Design Decisions

1. **Monorepo layout** — Annotation, extraction, and training share a `shared/` module for GCS and video utilities
2. **No model predictions shown to coders** — Prevents anchoring bias
3. **Per-coder annotation isolation** — All DB lookups/upserts keyed on `(video_id, annotator)`
4. **CSV output on GCS** — Simple, easy to inspect, compatible with pandas; persisted via GCS FUSE
5. **Feature CSVs keyed on video_id** — Each extractor outputs a CSV that joins with annotations
6. **Optional PyTorch** — torch/ModelLoader imports wrapped in try/except for cloud deployment without GPU

## Annotation Schema

| Field | Options |
|-------|---------|
| `video_id` | Unique video identifier |
| `annotator` | Coder name (entered at login) |
| `no_human_visible` | Boolean checkbox |
| `perspective` | "1st person", "2nd person", "3rd person", "NA" |
| `distance` | "Personal", "Social", "Public", "NA" |
| `notes` | Free text |
| `is_difficult` | Boolean flag |

## Feature Extraction

Run `python feature_extraction/extract_all.py` from the monorepo root.

| Extractor | Status | Key outputs |
|-----------|--------|-------------|
| Cut detection | Working | `cut_count`, `cuts_per_second`, `avg_scene_duration` |
| Density | Stub | `color_entropy`, `edge_density_ratio`, `avg_motion_magnitude` |
| Gaze | Stub | `num_faces`, `gaze_at_camera_ratio` |
| Object detection | Stub | `num_objects`, `num_humans`, `object_types` |
| Text detection | Stub | `has_text`, `text_area_ratio`, `text_changes_per_second` |

## Known Issues

- **Sidebar starts collapsed** despite `initial_sidebar_state="expanded"`. May be a Streamlit version issue or CSS interaction.
- **No concurrent write locking** on the shared CSV. Low risk at current team size.

## Bug Fixes (2026-02-08)

- **Duplicate annotations bug** — `pd.read_csv()` inferred `video_id` as int64, but the app passed it as a string. Fixed by casting `video_id` to `str` on every CSV read.
- **Auto-resume broken** — Same type mismatch. Fixed by same `astype(str)` cast.
- **Save & Next reliability** — Added spinner, retry, toast feedback.

## Dependencies

- Python 3.11+
- PyTorch (optional, with MPS support for Apple Silicon)
- Streamlit, OpenCV, pandas, numpy
- google-cloud-storage
- scenedetect (for cut detection)

## Contact

Project by Royce Choi for TikTok parasocial interaction research.
