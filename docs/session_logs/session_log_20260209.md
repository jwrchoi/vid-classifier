# Session Log — 2026-02-09

## Part 1: Monorepo Restructure

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

---

## Part 2: Feature Extraction Pipeline

### 5 extractors implemented and tested

All extractors follow the same pattern: `extract(video_path: Path, video_id: str) -> dict` returning a flat dictionary for CSV output. Each uses lazy-loaded models cached after first call.

| Extractor | Model/Library | Key Output Columns |
|-----------|--------------|-------------------|
| **cut_detection** | PySceneDetect ContentDetector | `cut_count`, `cuts_per_second`, `avg_scene_duration` |
| **density** | OpenCV (HSV histograms, Canny, Farneback optical flow) | `color_entropy`, `edge_density_ratio`, `avg_motion_magnitude` |
| **object_detection** | YOLOv8-nano (COCO 80 classes) | `avg_objects`, `num_humans`, `object_types` (JSON) |
| **text_detection** | EasyOCR (CRAFT + CRNN) | `has_text`, `text_area_ratio`, `text_changes_per_second` |
| **gaze** | MediaPipe FaceLandmarker (iris tracking) | `num_faces`, `gaze_at_camera_ratio`, `avg_gaze_yaw`, `avg_gaze_pitch` |

#### MediaPipe API migration

mediapipe 0.10.32 dropped the legacy `mp.solutions.face_mesh` API. Rewrote `gaze.py` to use the new `mp.tasks.vision.FaceLandmarker` with:
- Downloaded `face_landmarker.task` model file (auto-downloads if missing)
- Uses `mp.Image` input and `landmarker.detect()` instead of `FaceMesh.process()`
- 478 landmarks including 10 iris landmarks (5 per eye) for gaze estimation

#### Feature dictionary

Created `feature_extraction/docs/feature_dict.md` — full reference for all output columns, equations (Shannon entropy, edge density, optical flow magnitude, gaze ratios), and config parameters.

### Hardened `extract_all.py` for scale

Improvements for running on 17k+ videos on spot VMs:

| Feature | Description |
|---------|-------------|
| **Disk space guard** | Checks free space before each download; stops at < 1 GB to prevent corruption |
| **GCS checkpointing** | `--checkpoint-interval N` uploads CSVs to GCS every N videos; survives spot preemption |
| **Progress + ETA** | Shows remaining count and estimated completion time |
| **Safe temp cleanup** | `tmp_path` is `None`-checked in `finally` block (no crash if download fails) |
| **Summary report** | Success/fail/skip counts at end of run |

#### CSV path fix

`video_list_v2.csv` has a `gcs_path` column (e.g., `videos/01_filtered/video-xxx.mp4`) but the CSV reader fell through to the `video_id` fallback which constructed an incorrect path (`videos/{video_id}.mp4`). Added `gcs_path` as the first column to check.

---

## Part 3: GCE GPU VM (T4)

### Setup

Created a spot VM for GPU-accelerated feature extraction:

| Setting | Value |
|---------|-------|
| Name | `feature-extraction-gpu` |
| Machine | `n1-standard-8` (8 vCPU, 30 GB RAM) |
| GPU | 1x NVIDIA Tesla T4 (15 GB VRAM) |
| Provisioning | Spot (preemptible, ~$0.11/hr) |
| Disk | 100 GB SSD |
| Image | `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` |
| Zone | `us-central1-a` |
| Project | `vid-classifier` |

**Prerequisite:** Had to request `GPUS_ALL_REGIONS` quota increase from 0 → 1 via Cloud Console. Regional T4 quota was already 1, but the global cap blocked creation. Approved in minutes.

**Additional VM fix:** Installed `libgl1-mesa-glx` and `libglib2.0-0` (OpenCV dependencies missing from headless VM image).

### VM scripts

| Script | Purpose |
|--------|---------|
| `feature_extraction/scripts/create_gpu_vm.sh` | Creates the spot VM with all settings above |
| `feature_extraction/scripts/setup_vm.sh` | Clones repo, installs deps, verifies GPU/CUDA, downloads existing CSVs from GCS (resume support), runs smoke test |
| `feature_extraction/scripts/run_extraction.sh` | Runs extractors one at a time (fastest first), checkpoints every 25 videos to GCS, logs to per-extractor `.log` files |

### 50-video test run results

All 50 videos processed with zero failures. Run completed in **27 minutes 34 seconds**.

| Extractor | Time | Per-video avg | vs. local CPU |
|-----------|------|---------------|---------------|
| gaze | 1m 37s | 1.9s | ~same |
| cuts | 1m 12s | 1.4s | ~same (CPU-bound) |
| density | 12m 19s | 15.1s | ~2x slower (optical flow CPU-bound) |
| object_detection | 1m 28s | 1.8s | **3.3x faster** (GPU) |
| text_detection | 10m 48s | 13.0s | **2.8x faster** (GPU) |

All CSVs checkpointed to `gs://vid-classifier-db/features/`.

### Scaling estimate for 17,500 videos

| Extractor | Per-video | 17.5k total |
|-----------|-----------|-------------|
| gaze | 1.9s | ~9 hr |
| cuts | 1.4s | ~7 hr |
| density | 15s | ~73 hr |
| object_detection | 1.8s | ~9 hr |
| text_detection | 13s | ~63 hr |
| **Total** | **~33s** | **~161 hr (~6.7 days)** |

Density (optical flow) and text detection (OCR) are the bottlenecks — both CPU-bound even with GPU available. Two parallel VMs could cut total time to ~3.5 days.

**VM is currently stopped** (paused to avoid charges; disk-only cost ~$0.17/day). To resume:

```bash
gcloud compute instances start feature-extraction-gpu --project=vid-classifier --zone=us-central1-a
gcloud compute ssh feature-extraction-gpu --project=vid-classifier --zone=us-central1-a
# On VM:
cd ~/vid-classifier && git pull && source venv/bin/activate
bash feature_extraction/scripts/run_extraction.sh
```

---

## Commits

| Hash | Message |
|------|---------|
| `1e827f8` | Restructure to monorepo; add feature extraction pipeline |
| `c50dfb4` | Fix data path resolution after monorepo restructure |
| `bf0aade` | Implement all feature extractors: density, object detection, text, gaze |
| `358b5a3` | Add checkpointing, disk guard, GPU VM scripts, and feature dictionary |
| `cbf4d86` | Fix video list CSV parsing to use gcs_path column |

## Key files changed

- `annotation_dashboard/config.py` — Path resolution logic
- `annotation_dashboard/deploy.sh` — Seed data upload paths
- `shared/` — New shared utilities module
- `feature_extraction/extractors/*.py` — All 5 extractor implementations
- `feature_extraction/extract_all.py` — Orchestrator with checkpointing, disk guard, ETA
- `feature_extraction/config.py` — Settings for all extractors
- `feature_extraction/docs/feature_dict.md` — Feature dictionary
- `feature_extraction/scripts/*.sh` — VM creation, setup, and run scripts
- `CLAUDE.md` — Updated for monorepo structure
- `.gitignore` — Added YOLO/ML cache and `.task` model patterns

## Session log consolidation

Moved all session logs from `annotation_dashboard/docs/` to `docs/session_logs/` at the project root. Session logs cover work across the whole monorepo (restructuring, feature extraction, annotation dashboard fixes) and don't belong scoped to a single sub-project. Technical docs (like `feature_dict.md`) stay in their sub-project's `docs/` folder.
