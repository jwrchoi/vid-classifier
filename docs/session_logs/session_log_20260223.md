# Session Log — 2026-02-23

## Part 1: Inter-Coder Reliability (ICR) — Round 1 Analysis

### ICR queue generation (32 frames)

Soojin had annotated 32 unique `(video_id, frame_index)` pairs. To run an ICR check, a script was created to limit Janice's queue to exactly those 32 frames:

**`annotation_dashboard/scripts/generate_icr_queue.py`** — reads Soojin's annotated pairs from `gs://vid-classifier-db/annotations/annotations.csv`, writes a `queue.csv` with only those pairs, and uploads it to GCS. Because `get_effective_queue()` in `utils/queue.py` already checks for `queue.csv` before generating a shuffled queue, no dashboard code changes were needed.

```bash
python annotation_dashboard/scripts/generate_icr_queue.py --coder Soojin
# → Uploaded gs://vid-classifier-db/annotations/queue.csv (32 frames)
```

### ICR analysis script

**`annotation_dashboard/scripts/compute_icr.py`** — computes inter-coder reliability for `perspective` and `distance` on the overlapping set of frames coded by both coders.

Metrics reported:
- Percent agreement
- Cohen's kappa
- Krippendorff's alpha (nominal) — preferred for content analysis research; handles missing data natively; generalizes to >2 raters

When `no_human_visible=True`, both fields are treated as `"NA"` (valid label). Outputs an aligned wide-format CSV (`data/icr_report.csv`) for manual review and sharing.

```bash
python annotation_dashboard/scripts/compute_icr.py --coder1 Soojin --coder2 Janice
# → Saved data/icr_report.csv
```

### Round 1 ICR results (32 overlapping frames)

| Field | Frames with both coded | % Agreement | Cohen's κ | Krippendorff's α |
|-------|----------------------|-------------|-----------|-----------------|
| Perspective | 27 / 32 | 92.6% | 0.900 | 0.902 |
| Distance | 18 / 32 | 83.3% | 0.751 | 0.753 |

The reduced sample for distance (18 vs 27) is due to Soojin having blank distance values on 14 frames.

**True coder disagreements** (excluding blank-vs-label cases):
- Perspective: 2 frames (1st person vs 2nd person)
- Distance: 3 frames (Public vs Social)

---

## Part 2: Root-Cause Analysis — Soojin's Blank Annotations

### Why Soojin has blank perspective/distance

All 14 blank-distance frames (and 5 blank-perspective frames) share a pattern: `no_human_visible=False` but the field is empty. Root cause is the **form state bug fixed on 2026-02-09**: when a previous frame was coded as `no_human_visible=True`, the distance widget's stale session state carried over into the next frame. If the submit happened before the coder noticed, the distance was saved as empty string (not `"NA"`) because the fix deployed later that day may not have been active for all of Soojin's early sessions.

The 14 blank-distance / 5 blank-perspective frames are identified and queued for re-coding in Part 3.

---

## Part 3: ICR Round 2 — Per-Coder Queues

### Problem: Janice reported duplicate frames

**Root cause:** `load_queue_csv()` in `utils/queue.py` had no deduplication. If the upstream CSV (from the active learning pipeline or ICR scripts) contained duplicate rows, coders saw the same frame more than once.

**Fix (`utils/queue.py`):**
```python
df = df.drop_duplicates(subset=["video_id", "frame_index"])
```

### Per-coder queue lookup (`.title()` normalization)

**Problem:** The global `queue.csv` applied to all coders. ICR round 2 requires different queues per coder (Soojin needs her blank re-codes; Janice only needs the 30 new frames).

**Fix (`utils/queue.py`, `get_effective_queue()`):**

Added a per-coder queue lookup before the global fallback:
```
queue_{Annotator}.csv  →  per-coder (checked first, title-cased filename)
queue.csv              →  global fallback
(nothing)              →  full shuffled 500-frame queue
```

Annotator name is `.title()`-normalized for the filename lookup so that "soojin" and "Soojin" both resolve to `queue_Soojin.csv`.

### Blank field warning in annotation form

**Fix (`app.py`, `render_annotation_form()`):**

Added a render-time `st.warning` banner when the coder's existing annotation for the current frame has a blank perspective or distance (and `no_human_visible=False`). This makes incomplete frames self-evident without blocking navigation.

```
⚠️ Your previous save for this frame was incomplete —
Social Distance is missing. Please fill in all fields and save again.
```

### ICR v2 queue generation script

**`annotation_dashboard/scripts/generate_icr_v2_queue.py`** — end-to-end script for ICR round 2 setup:

1. Backs up `annotations.csv` on GCS
2. Finds Soojin's frames with blank perspective or distance (`no_human_visible=False`)
3. Samples 30 new `(video_id, frame_index)` pairs not yet coded by either coder (seed=42, reproducible)
4. Writes per-coder queue CSVs and uploads to GCS:
   - `queue_Soojin.csv` — 14 re-codes + 30 new = **44 frames**
   - `queue_Janice.csv` — 30 new frames = **30 frames**
5. Deletes the global `queue.csv` so other coders fall back to the full shuffled queue

The same 30 new frames appear in both queues (shared ICR set), in a different per-coder shuffle order. After both coders finish, re-run `compute_icr.py` for updated reliability numbers on the fresh 30-frame set.

```bash
python annotation_dashboard/scripts/generate_icr_v2_queue.py
# Soojin: reload dashboard → sees 44 frames (14 re-codes + 30 new)
# Janice: reload dashboard → sees 30 new frames
```

### GCS state after script

| Path | Status |
|------|--------|
| `annotations/queue_Soojin.csv` | Created (44 frames) |
| `annotations/queue_Janice.csv` | Created (30 frames) |
| `annotations/queue.csv` | Deleted |
| `annotations/backups/annotations_backup_20260223_000010.csv` | Created |

---

## Files Changed

| File | Change |
|------|--------|
| `annotation_dashboard/app.py` | Blank-field warning banner in annotation form |
| `annotation_dashboard/utils/queue.py` | Dedup in `load_queue_csv()`; per-coder queue lookup with `.title()` normalization in `get_effective_queue()` |
| `annotation_dashboard/scripts/generate_icr_queue.py` | New — generates 32-frame ICR queue from reference coder's annotations |
| `annotation_dashboard/scripts/compute_icr.py` | New — computes % agreement, Cohen's κ, Krippendorff's α; outputs `data/icr_report.csv` |
| `annotation_dashboard/scripts/generate_icr_v2_queue.py` | New — generates ICR v2 per-coder queues; backs up, samples 30 new frames, uploads, deletes global queue |

## Deployed

App redeployed to Cloud Run after `app.py` and `queue.py` changes.
Coders need to hard-reload (`Cmd+Shift+R`) to pick up the new queue from GCS.
