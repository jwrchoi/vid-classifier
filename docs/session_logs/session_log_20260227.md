# Session Log — 2026-02-27

## Data Analysis Pipeline Setup

### Objective

Build an analysis pipeline from extracted visual features to engagement outcomes. Unit of analysis: individual TikTok video. Goal: establish how CV-extracted visual features predict engagement metrics (views, likes, comments, shares, bookmarks).

---

## Part 1: Dataset Assessment

### GCS feature inventory

All 5 feature extractors completed on the full video corpus:

| Extractor | GCS file | Rows | Unique videos |
|-----------|----------|------|---------------|
| Cut detection | `features/cuts.csv` | 22,186 | 15,202 |
| Density | `features/density.csv` | 29,322 | 15,231 |
| Gaze | `features/gaze.csv` | 23,149 | 15,231 |
| Object detection | `features/object_detection.csv` | 23,108 | 15,231 |
| Text detection | `features/text_detection.csv` | 28,888 | 15,231 |

Duplicate rows per video (from checkpoint resumption) were cleaned via `drop_duplicates(subset="video_id", keep="last")`.

### Metadata

`data/video_metadata/02_cleaned_metadata/metadata_final_english.csv` — 15,614 videos with engagement columns (`views`, `likes`, `comments`, `shares`, `bookmarks`), `channel` (JSON string), `uploadedAt` (Unix timestamp), `keyword`.

### Final merged dataset

**15,202 videos × 44 columns** after inner-joining all 5 feature tables with metadata.

Key data notes:
- `avg_gaze_pitch` has extreme outliers (min −1875, max +1860) — likely extraction artifacts on tilted faces. Winsorized at p1/p99 before modeling.
- `avg_gaze_yaw` similarly has extreme outliers. Both excluded from modeling in favor of `gaze_at_camera_ratio`.
- Engagement is heavily right-skewed (median views ~4K, max ~92M). Log₁p transformation applied.

---

## Part 2: `analysis/build_merged_dataset.py`

Standalone Python script that produces `analysis/merged_dataset.csv`. Key steps:

1. **Load + deduplicate** each feature CSV (keep last checkpoint row per video)
2. **Inner-join** all 5 feature tables, then join with metadata
3. **Parse `channel` JSON** → extract `creator_username`, `creator_followers`, `creator_verified`; compute `log_creator_followers`
4. **Parse `uploadedAt`** Unix timestamp → `upload_date`, `time_trend_days` (continuous, 0–2619), `upload_month` (categorical "01"–"12"), `upload_year`
5. **Extract brand** from `keyword` using pattern matching → `brand` column (OnRunning 5,110; Brooks 3,341; Hoka 2,508; Saucony 2,374; Other 1,869)
6. **Add `log1p` engagement columns**: `log_views`, `log_likes`, `log_comments`, `log_shares`, `log_bookmarks`

Run: `python3 analysis/build_merged_dataset.py`

---

## Part 3: `analysis/01_eda.ipynb` — Exploratory Data Analysis

7-section EDA notebook:

| Section | Content |
|---------|---------|
| 1 — Overview | Shape, dtypes, missing value chart |
| 2 — Engagement distributions | Raw histograms + log-transformed; skewness table |
| 3 — Feature distributions | Histograms + boxplots for all 20 visual features |
| 4 — Anomaly detection | 17 rule-based flags (fps=0, gaze extremes, duration outliers, etc.) |
| 5 — Correlation analysis | Full heatmap; bar chart of r with log_views |
| 6 — Scatter plots | Top 6 feature × log_views regplots |
| 7 — Summary table | Top 10 correlates with log_views ranked by \|r\| |

**Notable EDA findings:**
- Strongest raw correlates with `log_views`: `cut_count` (r ≈ 0.08), `avg_motion_magnitude`, `edge_density_ratio` — all modest
- Visual features alone show weak linear correlations; engagement is driven more by creator reach

---

## Part 4: `analysis/02_modeling.ipynb` — Regression Modeling

Follows the reference notebook style (statsmodels OLS, Patsy formulas, HC3 robust SEs, standardized predictors). Adapted for this project's feature set.

### Visual predictors (all standardized, _z suffix)

`cuts_per_second`, `color_entropy`, `edge_density_ratio`, `avg_motion_magnitude`, `gaze_at_camera_ratio`, `num_faces`, `avg_objects`, `has_text` (binary, not standardized), `text_area_ratio`, `text_changes_per_second`

### Control variables

`duration_sec_z`, `log_creator_followers_z` (strongest confounder), `time_trend_days_z`, `creator_verified`, `C(brand, Treatment('Other'))`, `C(upload_month)`

### Engagement DVs

`log_views`, `log_likes`, `log_comments`, `log_shares`, `log_bookmarks`

### Pipeline

1. Winsorize all continuous features at p1/p99
2. Z-score standardize continuous predictors
3. Fit OLS with HC3 robust SEs for all 5 DVs
4. VIF diagnostics on visual predictors
5. Outlier detection via standardized residuals (\|z\| > 3.5) → 175 rows removed
6. Clean re-run on 15,027 rows with residual diagnostics
7. Random Forest feature importance (visual predictors only)
8. Export to `analysis/model_exports/` (CSV, HTML, LaTeX)

### Preliminary results (log_views, clean sample)

- OLS R² = 0.364 — good for social media data; driven primarily by `log_creator_followers`
- RF CV R² (visual features only, no controls) = 0.008 — confirms visual features have modest standalone predictive power
- RF feature importance ranking: `edge_density_ratio` > `avg_motion_magnitude` > `text_area_ratio` > `color_entropy` > `avg_objects` > `text_changes_per_second`

### Improvements over reference notebook

| Issue in reference | Fix in our notebook |
|---|---|
| `ValueError: too many values to unpack` — studentized residual loop unpacked 3-tuple keys as 2-tuple | `ols_results` keyed by plain string target; no tuple unpacking |
| `"P>|z|"` column rename (wrong for OLS) | Uses `"P>|t|"` with fallback search for actual column name |
| No winsorization before modeling | Winsorize cell runs before standardization |
| `OLSInfluence.resid_studentized_external` is O(n²) — timed out on 15K rows | Replaced with fast standardized residuals (`resid / sqrt(MSE)`); equivalent at large n |
| No non-parametric benchmark | Random Forest feature importance section added |

---

---

## Part 5: `analysis/02_modeling.ipynb` — Interactions, GBM, LASSO (Sections 8–10)

### Section 8 — OLS Interaction Terms

8 theoretically motivated pairs screened on `log_views`; 4 significant at p < .05:

| Interaction | β (log_views) | Key finding |
|---|---|---|
| `gaze_at_camera_ratio × num_faces` | +0.098* | Strongest cross-DV finding: β = +0.21*** for shares, +0.19*** for bookmarks — direct gaze drives saves/shares specifically |
| `cuts_per_second × avg_motion_magnitude` | −0.047* | Kinetic overload; significant for likes and bookmarks too |
| `color_entropy × edge_density_ratio` | −0.044* | Compounding visual complexity is negative; uniquely significant for comments** |
| `text_area_ratio × cuts_per_second` | −0.039* | Information overload; significant for bookmarks* |

Adj. R² gains modest (+0.0007 for log_views; +0.0017 for shares and bookmarks).
Exported: `analysis/model_exports/ols_interactions.{csv,html,tex}`

### Section 9 — Gradient Boosting Machine (GBM)

| Model | R² |
|---|---|
| RF visual-only (CV) | 0.008 |
| GBM visual-only (CV) | 0.035 |
| OLS visual + controls (adj.) | 0.363 |
| GBM visual + controls (CV) | 0.447 |

- GBM extracts 4.5× more signal from visual features than RF — signal is non-linear/interactive
- GBM with controls outperforms OLS by ~0.08 R² — confirms genuine non-linearity
- Permutation importance: `log_creator_followers` dominates (0.216); visual features with positive importance: `text_changes_per_second` > `edge_density_ratio` > `avg_motion_magnitude` > `avg_objects`
- Features with negative permutation importance (overfit/context-dependent): `num_faces`, `gaze_at_camera_ratio`, `text_area_ratio`, `cuts_per_second`, `has_text`, `color_entropy`

### Section 10 — LASSO & Model Comparison

- LASSO α = 0.0012; train R² = 0.341; all 19 features survive regularization → none redundant enough to zero out
- `log_creator_followers` LASSO coef = 1.34 (dominant); top visual coef: `text_changes_per_second` (+0.088), `edge_density_ratio` (+0.053), `avg_motion_magnitude` (+0.038)

### Alternative Approaches (documented in Section 10c)

Described with implementation notes: Linear Mixed Model (creator as random effect), Negative Binomial on raw counts, Quantile regression (75th/90th percentile), Two-stage Heckman, LightGBM/XGBoost, Time-series CV.

Recommended next steps: LMM with creator random effect; quantile regression at 75th/90th; GBM 2D PDP on top interaction.

---

## Files Changed

| File | Change |
|------|--------|
| `analysis/build_merged_dataset.py` | New — merges 5 feature CSVs + metadata; parses channel JSON, timestamps, brand |
| `analysis/01_eda.ipynb` | New — 7-section EDA notebook (gitignored, run locally) |
| `analysis/02_modeling.ipynb` | New — OLS + RF + interactions + GBM + LASSO (gitignored, run locally) |
| `analysis/model_exports/ols_results.{csv,html,tex}` | New — baseline OLS regression tables |
| `analysis/model_exports/ols_interactions.{csv,html,tex}` | New — interaction model regression tables |
