# Session Log — 2026-02-20

## Fix: Legacy annotations + annotation editing UX

### Problem 1: Legacy annotations causing false "already annotated"

Legacy video-level annotations (from the pre-frame-level dashboard) had `frame_index=NaN`. While pandas `NaN != float(X)` comparisons usually exclude these rows, edge cases could cause false matches in `get_annotation()`.

**Fix (database.py):**
- Added explicit `df['frame_index'].notna()` guard in `get_annotation()` so NaN rows can never match a frame lookup
- Added `purge_legacy_rows()` method that backs up `annotations.csv` then drops all rows where `frame_index` is NaN
- Purge runs once per session at startup during initialization

### Problem 2: Coders think annotated frames are locked

The dashboard showed `st.success("You have already annotated this frame.")` when revisiting a frame. Coders interpreted this as "locked" and didn't realize the form below was editable.

**Fix (app.py):**
- Changed to `st.info("You've annotated this frame. You can update your response below.")` to clarify editing is possible

### Problem 3: Social Distance locked to NA on revisited frames

When revisiting a frame previously marked "no human visible", the Social Distance section showed only "Automatically set to NA" with no radio buttons. Because the checkbox is inside a `st.form`, unchecking it doesn't trigger a rerun — the coder had no way to see or change the distance options.

**Fix (app.py):**
- Distance radio buttons are now always visible regardless of the checkbox state
- When "No human visible" is checked, a caption explains the value will be saved as NA and the radio defaults to NA
- NA override enforced at save time (`'NA' if no_human_visible else distance`) so the business logic is preserved

### Bug fix: deploy.sh running from wrong directory

`gcloud builds submit` requires a Dockerfile in the build context directory, but the script didn't specify the source directory — it used CWD, which fails when run from the repo root.

**Fix (deploy.sh):**
- Changed `gcloud builds submit` to explicitly pass `${SCRIPT_DIR}` as the build context

---

## Files changed

- `annotation_dashboard/utils/database.py` — `notna()` guard in `get_annotation()`, new `purge_legacy_rows()` method
- `annotation_dashboard/app.py` — UX message change, always-visible distance radio, NA override at save time, legacy purge at startup
- `annotation_dashboard/deploy.sh` — Explicit build context directory for `gcloud builds submit`

## Deployed

App redeployed to Cloud Run: `https://shoe-annotator-302331461875.us-central1.run.app`
