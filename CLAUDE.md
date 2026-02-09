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
| Videos | Streamed from GCS bucket `vid-classifier-db` (prefix `videos/01_filtered/`) |
| Trained Models | `/Users/roycechoi/proj/tiktok_analysis/codes/artifacts/model_exports/resnet50_models/` |
| Annotations Output | `./data/annotations.csv` (GCS FUSE-mounted on Cloud Run) |
| Video List | `./data/video_list_v2.csv` |

## Models (v7)

- `pov_resnet_50_v7.pth` - Predicts: 1st person, 2nd person, 3rd person
- `social_distance_resnet_50_v7.pth` - Predicts: Personal, Social, Public

Models are optional — the app runs without PyTorch installed (torch/ModelLoader imports are wrapped in try/except).

## Architecture

```
running_shoe_vid_classifier/
├── app.py                  # Main Streamlit dashboard
├── config.py               # Configuration (paths, models, device)
├── CLAUDE.md               # This file
├── Dockerfile              # Container image for Cloud Run
├── .dockerignore           # Excludes venv, data, models from build
├── deploy.sh               # End-to-end GCP deployment script
├── models/
│   ├── model_loader.py     # ResNet-50 loading & inference
│   └── feature_extractors.py  # (Not used in current UI)
├── utils/
│   ├── video_processing.py # Frame sampling
│   ├── gcs.py              # GCS video streaming
│   └── database.py         # CSV annotation storage (per-coder)
├── scripts/
│   └── build_filtered_video_list.py  # Generate filtered video list CSV
├── batch_preprocess.py     # Pre-compute predictions (optional)
├── requirements.txt        # Dependencies
├── docs/
│   ├── session_log_20250207.md  # Session changelog
│   └── session_log_20260208.md  # Duplicate annotation bug fix
└── data/                   # Created at runtime (GCS FUSE on Cloud Run)
    ├── annotations.csv     # Human annotations output
    └── video_list_v2.csv   # Fixed video list
```

## Running the App

```bash
# Local development
source venv/bin/activate
streamlit run app.py

# Deploy to Cloud Run
bash deploy.sh
```

## Deployment (Cloud Run)

`deploy.sh` handles everything end-to-end:
1. Enables GCP APIs (Cloud Run, Cloud Build, Artifact Registry, Storage)
2. Creates Artifact Registry repo and service account
3. Uploads seed data to GCS
4. Builds Docker image via Cloud Build
5. Deploys to Cloud Run with GCS FUSE volume mount

The GCS bucket `vid-classifier-db` is mounted at `/app/data`, so `annotations.csv` is read/written directly to GCS with no separate sync step.

## Key Design Decisions

1. **No model predictions shown to coders** - Prevents anchoring bias
2. **"No human visible" checkbox** - Screener question; affects which annotation fields are applicable
3. **Multi-category options** - Not binary; includes NA for edge cases
4. **CSV output on GCS** - Simple, easy to inspect, compatible with pandas; persisted via GCS FUSE
5. **Per-coder annotation isolation** - All DB lookups/upserts keyed on `(video_id, annotator)` so coders don't overwrite each other
6. **Auto-resume on login** - On init, jumps coder to their first unannotated video
7. **Side-by-side layout** - Video left (~40%), annotation form right (~60%) via `st.columns([2, 3])`; auto-stacks vertically on narrow screens
8. **Optional PyTorch** - torch/ModelLoader imports wrapped in try/except for cloud deployment without GPU

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

## Common Tasks

### Add a new annotator
Just enter name on login screen - no setup needed. Each coder gets independent annotation rows.

### Export annotations
Annotations auto-save to `data/annotations.csv`. On Cloud Run, this is the GCS-mounted file.

### Pre-process videos (optional, for speed)
```bash
python batch_preprocess.py --limit 50
```

## Known Issues

- **Sidebar starts collapsed** despite `initial_sidebar_state="expanded"`. May be a Streamlit version issue or CSS interaction. The collapse button is styled to be always visible.
- **No concurrent write locking** on the shared CSV. Low risk at current team size but could lose writes if two coders submit at the exact same millisecond.

## Bug Fixes (2026-02-08)

- **Duplicate annotations bug** — `pd.read_csv()` inferred `video_id` as int64, but the app passed it as a string. The type mismatch caused upsert checks to always fail, appending duplicate rows instead of updating. Fixed by casting `video_id` to `str` on every CSV read in `utils/database.py`. Also changed `save_annotation` to delete-then-append (guarantees one row per video per coder) and `get_annotation_stats` to count unique video_ids instead of total rows.
- **Auto-resume broken** — Same type mismatch caused `get_annotated_video_ids()` to return int IDs that never matched the string IDs in the video list, so coders always started at video #1. Fixed by the same `astype(str)` cast.
- **Deduped existing data** — Ran one-time dedup on GCS `annotations.csv` (16 → 14 rows, removed 2 Soojin duplicates).
- **Save & Next reliability** — Added spinner during save, retry on transient GCS failure, toast confirmation that persists across rerun, and explicit error if db is uninitialized. Previously a failed save silently left the video unchanged with no feedback.

## Dependencies

- Python 3.11+
- PyTorch (optional, with MPS support for Apple Silicon)
- Streamlit
- OpenCV
- pandas, numpy
- google-cloud-storage
- langdetect

## Contact

Project by Royce Choi for TikTok parasocial interaction research.
