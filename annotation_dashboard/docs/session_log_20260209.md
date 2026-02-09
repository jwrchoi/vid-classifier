# Session Log — 2026-02-09

## Monorepo Restructure & Feature Extraction Pipeline

### What was done

#### 1. Restructured project into monorepo

Moved the flat `running_shoe_vid_classifier/` project into a monorepo layout under `tiktok_video_analysis/`:

- `annotation_dashboard/` — All existing dashboard code (`app.py`, `config.py`, `models/`, `utils/`, `scripts/`, `Dockerfile`, `deploy.sh`)
- `feature_extraction/` — New CV feature extraction pipeline
- `model_training/` — Placeholder for future fine-tuning scripts
- `shared/` — Reusable GCS and video utilities shared across sub-projects
- `data/` — Stays at repo root (GCS FUSE-mounted on Cloud Run)

Used `git mv` to preserve full commit history for all relocated files.

#### 2. Fixed broken paths after restructure

`config.py` used `Path("data")` (CWD-relative), which broke when the app moved into `annotation_dashboard/` — the dashboard showed a blank screen because it couldn't find the video list or database.

**Fix:** Anchored all paths to `__file__` location:
- `_APP_DIR = Path(__file__).resolve().parent` (annotation_dashboard/)
- `_REPO_ROOT = _APP_DIR.parent` (tiktok_video_analysis/)
- `OUTPUT_DIR` checks `_REPO_ROOT/data` first (local dev), falls back to `_APP_DIR/data` (Cloud Run)

Also fixed:
- `deploy.sh` seed data upload paths (`data/` → `../data/` relative to script)
- `MODELS_DIR_CANDIDATES` — switched from CWD-relative to `__file__`-relative

#### 3. Recreated virtualenv

The venv was originally created at `~/proj/running_shoe_vid_classifier/venv/`. After the folder rename to `tiktok_video_analysis/`, all shebang lines in venv scripts (`streamlit`, `pip`, etc.) pointed to the old path. Recreated the venv in place and reinstalled all 70 packages from a saved `pip freeze`.

#### 4. Added feature extraction pipeline

- `feature_extraction/extract_all.py` — Orchestrator: downloads videos from GCS, runs extractors, writes per-extractor CSVs
- `feature_extraction/extractors/cut_detection.py` — Working implementation using PySceneDetect ContentDetector
- Stub extractors for density, gaze, object detection, text detection (docstrings describe planned approach)
- `shared/` module with `gcs_utils.py`, `video_utils.py`, `config.py`

#### 5. Git & deployment

- Pushed both commits to `origin/main` (github.com/jwrchoi/vid-classifier)
- Current Cloud Run deployment is unaffected (uses existing container image)
- To redeploy: run `bash deploy.sh` from `annotation_dashboard/`
- No repo rename needed — local folder name doesn't need to match GitHub repo name

### Commits

| Hash | Message |
|------|---------|
| `1e827f8` | Restructure to monorepo; add feature extraction pipeline |
| `c50dfb4` | Fix data path resolution after monorepo restructure |

### Key files changed

- `annotation_dashboard/config.py` — Path resolution logic
- `annotation_dashboard/deploy.sh` — Seed data upload paths
- `shared/` — New shared utilities module
- `feature_extraction/` — New extraction pipeline
- `CLAUDE.md` — Updated for monorepo structure
- `.gitignore` — Added YOLO/ML cache patterns
