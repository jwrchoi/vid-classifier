# Session Log — 2026-03-01

## Part 1: Fine-tune ResNet-50 v7 → v8 with LOOCV

### New script: `model_training/fine_tune_icr.py`

Fine-tunes v7 ResNet-50 models on agreed ICR frames using Leave-One-Out Cross-Validation (LOOCV), then compares v7 vs v8 accuracy. Only saves v8 weights if they improve.

**Design decisions for small-data regime (~40 samples):**
- LOOCV instead of train/val/test split (N too small)
- Freeze layers 1–3, only train `layer4` + `fc` (~8M of 23M params)
- Low learning rate: 1e-5 (10x lower than default)
- Inverse-frequency class weights for imbalanced classes
- 15 max epochs, patience=3 on training loss

**Data pipeline:**
1. Downloads `annotations.csv` from GCS (not the stale local copy)
2. Extracts doubly-coded frames where both coders agree (the "gold standard" set)
3. Preloads all frames into memory to avoid re-downloading per LOOCV fold
4. Runs LOOCV for both POV and distance

### LOOCV Results

**Perspective (POV) — 45 agreed frames:**

| Metric | v7 | v8 | Delta |
|--------|----|----|-------|
| Accuracy | 0.3111 | 0.4000 | +8.9% |
| Macro F1 | 0.1795 | 0.3016 | +12.2% |

- 4 folds fixed by v8, 0 regressions
- v7 has heavy "1st person" bias — predicts it for nearly everything
- v8 recovers some 3rd person predictions

**Distance — 39 agreed frames:**

| Metric | v7 | v8 | Delta |
|--------|----|----|-------|
| Accuracy | 0.4872 | 0.5897 | +10.3% |
| Macro F1 | 0.5247 | 0.6163 | +9.2% |

- 5 folds fixed, 1 regression
- v7 was already decent; v8 adds net +4 correct predictions

Both models improved, so both v8 weights were saved.

### Config change

Added `V8_OUTPUT_DIR = ROUNDS_DIR / "v8"` to `model_training/config.py`.

### Outputs (gitignored)

- `model_training/rounds/v8/v8_comparison.json`
- `model_training/rounds/v8/pov_resnet50_v8.pth`
- `model_training/rounds/v8/distance_resnet50_v8.pth`

---

## Part 2: Batch Inference Pipeline for 15K Videos

### New script: `model_training/batch_inference.py`

Producer/consumer architecture for running both models on all ~15,230 videos in GCS:

```
[GCS video list] → [N download workers (multiprocessing.Pool)] → [GPU inference (main process)]
```

**Key features:**
- 4 download worker processes (I/O-bound) feeding a single GPU consumer
- Checkpoints to GCS every 500 videos (spot VM resilience)
- Resumes from existing CSV if interrupted
- Outputs frame-level and video-level (majority vote) prediction CSVs

**Output CSVs:**
- `data/features/model_predictions.csv` — per-frame predictions with confidence + full probability vectors
- `data/features/model_predictions_video.csv` — per-video majority vote aggregation

### New script: `model_training/scripts/run_inference.sh`

End-to-end VM orchestration:
1. Start the GCE T4 GPU VM
2. Upload model weights via SCP
3. SSH in, clone/pull repo, install deps, run inference
4. Download results
5. Stop VM

---

## Part 3: Inter-Coder Reliability (ICR) Report

Ran `compute_icr.py` on the current annotations (120 rows, 53 overlapping frames).

**Perspective:**
- Frames with both coded: 50/53
- Percent agreement: 94.0%
- Cohen's kappa: 0.916
- Krippendorff's alpha: 0.917

**Distance:**
- Frames with both coded: 41/53
- Percent agreement: 85.4%
- Cohen's kappa: 0.797
- Krippendorff's alpha: 0.798

Of the 18 distance disagreements, 12 are Soojin's blank annotations from the legacy form-state bug (already queued for re-coding). The 6 genuine disagreements are mostly Personal/Social and Public/Social boundary cases.

---

## Bug fix

`fine_tune_icr.py` initially preferred the local `data/annotations.csv` (13 test rows) over GCS (120 real annotations). Fixed to always download from GCS first, falling back to local only if GCS fails.

---

## Commits

| Hash | Message |
|------|---------|
| (this session) | Add v8 fine-tuning with LOOCV and batch inference pipeline |

## Key files changed/created

- `model_training/config.py` — Added `V8_OUTPUT_DIR` path constant
- `model_training/fine_tune_icr.py` — **New** — LOOCV fine-tuning script
- `model_training/batch_inference.py` — **New** — Parallel inference on 15K videos
- `model_training/scripts/run_inference.sh` — **New** — GCE VM orchestration script
- `docs/session_logs/session_log_20260301.md` — This file

---

## Part 4: Modeling Notebook — Interactions, GBM PDPs, Alternative Count Models

Extended `analysis/02_modeling.ipynb` with focal interaction analysis, GBM partial dependence plots, and Negative Binomial / Poisson regressions.

### Removed `log_bookmarks` as a DV

All models now predict 4 outcomes: views, likes, comments, shares. Bookmarks dropped per research decision.

### Section 8a–b: Focal three-way interactions (theory-driven)

Tested all combinations of **color_entropy × avg_objects × C**, where C is:
1. `text_area_ratio` (text size)
2. `text_changes_per_second` (text pace)
3. `gaze_at_camera_ratio` (parasocial cue)

`has_text` was excluded — redundant with the continuous text measures, which are already 0 when no text is present.

Each model includes all lower-order two-way terms + the three-way term, fitted across all 4 DVs (12 models total). Results compiled and exported to:
- `analysis/focal_interactions_full.csv` (48 rows, every term × DV)
- `analysis/focal_interactions_pivot.csv` (12 rows, pivoted by DV)

**Key finding:** `color_entropy × avg_objects × text_area_ratio` is the only significant three-way term (p = .042 for views). Negative coefficient = "visual overload" penalty when all three are simultaneously high. The two-way `color_entropy × text_area_ratio` is more consistently significant across DVs.

### Section 8b-ii: Simple slopes plots

Added simple slopes visualization for all significant (p < .05) focal interaction terms:
- **Two-way**: effect of predictor A at -1 SD, mean, +1 SD of predictor B
- **Three-way**: three panels showing how the A×B slope changes at low/mean/high C

### Section 8b-iii: OLS interaction export

Full model output (all coefficients including controls) exported to:
- `analysis/model_exports/ols_interactions.csv/html/tex` — focal interaction models (408 rows)
- `analysis/model_exports/ols_pairwise_interactions.csv/html/tex` — data-driven pairwise interactions (separate file, no longer overwrites focal)

### Section 8c–f: Data-driven pairwise screening (relabeled)

Existing pairwise interaction screening (gaze × num_faces, etc.) preserved but renumbered from 8a→8c through 8d→8f to come after the focal interactions.

### Section 9e: 2D GBM Partial Dependence plots

Contour heatmaps for 5 focal pairs: color_entropy × avg_objects, color_entropy × gaze, color_entropy × text_area_ratio, color_entropy × text_changes_per_second, avg_objects × gaze.

### Section 9f: 3-Way GBM Partial Dependence (sliced contours)

For each of the 3 focal triples, the third variable is sliced at Q25/Q50/Q75 and a 2D PDP contour is drawn at each slice. Shows how the interaction surface shifts across levels of the third variable.

### Section 11: Alternative count models (NB, Poisson)

- **11a**: Overdispersion diagnostics — var/mean ratios are extreme (views: 9.4M, likes: 773K)
- **11b**: Poisson vs NB for views — Poisson emphatically rejected (AIC 12.6B vs NB 334K)
- **11c**: Log-OLS vs NB coefficient comparison — all directions agree; NB additionally finds `has_text` significant (p = .02)
- **11d**: NB for all 4 DVs — NB preferred for views/likes/comments; shares failed to converge (24% zeros, may need ZINB)
- **11e**: Guidance — log-OLS defensible as primary spec when NB gives same substantive story; report NB as robustness check

### Model comparison (Section 10b)

| Model | Metric | Value |
|-------|--------|-------|
| OLS (visual + controls) | adj. R² | 0.3615 |
| OLS (+ interactions) | adj. R² | 0.3623 |
| GBM (visual + controls) | CV R² | 0.4449 |
| LASSO (visual + controls) | train R² | 0.3398 |

---

## Key files changed/created (Part 4)

- `analysis/02_modeling.ipynb` — Extended with Sections 8a–b (focal interactions + simple slopes + export), 9e–f (2D/3D PDPs), 11a–e (NB/Poisson)
- `analysis/model_exports/ols_interactions.csv/html/tex` — **New** — Focal interaction OLS results
- `analysis/model_exports/ols_pairwise_interactions.csv/html/tex` — **New** — Data-driven pairwise interaction results (renamed from ols_interactions)
- `analysis/focal_interactions_full.csv` — **New** — All focal interaction terms × DVs
- `analysis/focal_interactions_pivot.csv` — **New** — Pivoted summary by DV
