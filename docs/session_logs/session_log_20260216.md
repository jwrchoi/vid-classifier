# Session Log — 2026-02-16

## Randomized Frame Presentation + Active Learning Pipeline

### Problem

1. **Consistency bias**: Frames were presented grouped by video with a thumbnail strip and "Video #X" header, so coders could anchor on earlier frames when labeling later ones from the same video.
2. **No active learning**: After the initial 50-video seed annotation, there was no automated way to scale — train models on collected labels, score unlabeled frames by uncertainty, and select the most informative frames for the next batch.

### What was done

#### Part A: Randomized Frame Presentation

- **New `annotation_dashboard/utils/queue.py`** — Queue generation and management. Each coder gets a deterministic but unique frame order seeded by `md5(annotator + salt)`. If an active-learning queue CSV exists, it takes priority.
- **Updated `annotation_dashboard/config.py`** — Added `QUEUE_CSV_PATH`, `QUEUE_SEED_SALT`, `FRAMES_PER_VIDEO`.
- **Updated `annotation_dashboard/utils/database.py`** — Added `get_all_annotated_pairs(annotator)` for efficient resume lookup.
- **Updated `annotation_dashboard/utils/gcs.py`** — Added `fetch_video_frames()` to download + extract frames as PNG bytes (Streamlit-cached).
- **Rewrote `annotation_dashboard/app.py`** — Replaced video/frame index navigation with queue-based state. Single frame display with no thumbnail strip, no video header, no filename. Simplified sidebar with single progress bar and Prev/Next on queue position. Frame cache and prefetching for upcoming videos.

#### Part B: Active Learning Pipeline

- **New `model_training/config.py`** — Training paths (rounds dir, frame cache, dashboard queue), model configs for perspective (3 classes) and distance (3 classes), training defaults (epochs=30, batch_size=32, lr=1e-4, patience=5), image preprocessing constants matching inference.
- **New `model_training/dataset.py`** — `FrameDataset` (PyTorch Dataset) that loads frames by `(video_id, frame_index)` with local PNG caching from GCS. Train transform with augmentation (flip, rotation, color jitter); eval transform with resize + normalize only.
- **New `model_training/trainer.py`** — `build_model()` creates ResNet-50 matching `ModelLoader` architecture exactly. `train_model()` with Adam optimizer, StepLR scheduler, early stopping on val loss. `evaluate()` with accuracy, macro F1, per-class metrics, confusion matrix via sklearn.
- **New `model_training/uncertainty.py`** — Scores unlabeled frames by combined prediction entropy (`H_pov + H_dist`). High combined entropy = most informative to annotate next.
- **New `model_training/reliability.py`** — Cohen's kappa on doubly-coded `(video_id, frame_index)` pairs for inter-coder agreement.
- **New `model_training/active_learning.py`** — CLI orchestrator for round-based active learning:
  1. Load annotations
  2. Round 0: create fixed test set (stratified 20%, saved to `rounds/test_set.csv`, never modified)
  3. Split remaining into train/val
  4. Compute inter-coder kappa
  5. Resolve starting weights (round 0 → v7 models; round N → round N-1 best)
  6. Train POV + distance models
  7. Score unlabeled pool by entropy, select top-K
  8. Write `round_NN/queue.csv` + `data/queue.csv` (dashboard picks this up)
  9. Save `round_NN/metrics.json` with stopping guidance
- **Updated `model_training/fine_tune_pov.py` and `fine_tune_distance.py`** — Replaced `NotImplementedError` with working wrappers around `trainer.train_model()`.
- **Updated `model_training/requirements.txt`** — Added matplotlib, tqdm, Pillow.
- **Updated `.gitignore`** — Added `model_training/frame_cache/` and `model_training/rounds/`.

### Key design decisions

- **Weight initialization chain**: Round 0 starts from v7 models. Each subsequent round starts from the previous round's best checkpoint.
- **Fixed test set**: Created once in round 0, never modified — gives unbiased performance estimates across rounds.
- **Per-coder shuffle**: md5-seeded so each coder gets a different but reproducible order.
- **Combined entropy**: Uncertain on either POV or distance = informative frame.
- **Stopping criteria**: Built into the orchestrator — flags diminishing returns (<1% F1 improvement) and overfitting (>15% train-val gap).

### Usage

```bash
# Dashboard (frames now randomized, no video grouping)
cd annotation_dashboard && streamlit run app.py

# Active learning round 0 (after seed annotations collected)
python model_training/active_learning.py --round 0 --top-k 50

# Subsequent rounds
python model_training/active_learning.py --round 1 --epochs 30
```

---

## Feature Extraction: Full Dataset Deployment

### Problem
Feature extraction was only tested on a 50-video sample. Need to run all 5 extractors against the full 15,231 filtered videos on a GCE GPU VM.

### What was done

- **`shared/config.py`** — Changed `GCS_VIDEO_PREFIX` from `"videos/"` to `"videos/01_filtered/"` to target the curated dataset (filtered against `metadata_final_english.csv`), avoiding duplicates from `00_videos/`.
- **`feature_extraction/scripts/run_extraction.sh`** — Rewrote for parallel execution. All 5 extractors now launch as concurrent background processes (CPU extractors run alongside the GPU-bound `text_detection`). Dropped `--video-list` flag so the pipeline scans GCS directly. Estimated wall time reduced from ~8 days to ~6.5 days.
- **`feature_extraction/scripts/create_gpu_vm.sh`** — Switched from spot to standard provisioning. Removed `--provisioning-model=SPOT`, `--instance-termination-action=STOP`, and `--no-restart-on-failure` since a multi-day job on spot would almost certainly get preempted.

### GCS bucket structure
- `videos/00_videos/` — 18,393 raw downloaded TikTok videos
- `videos/01_filtered/` — 15,231 videos filtered against English-language metadata

### Deployment steps
1. Delete old spot VM, recreate as standard
2. SSH in, run `setup_vm.sh`
3. Start extraction in tmux: `bash feature_extraction/scripts/run_extraction.sh`
4. Detach tmux — job runs unattended for ~6.5 days
5. Monitor via `tail -f data/features/<extractor>.log` or check GCS checkpoints
