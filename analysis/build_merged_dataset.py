"""
build_merged_dataset.py
-----------------------
Merges all extracted feature CSVs with TikTok video metadata into a single
analysis-ready dataset.

Steps:
  1. Load each feature CSV, deduplicate by video_id (keep last checkpoint row).
  2. Load metadata CSV (id -> video_id).
  3. Inner-join all 5 feature tables on video_id.
  4. Inner-join with metadata on video_id.
  5. Drop non-numeric / QC columns (object_types, num_frames_sampled variants).
  6. Parse channel JSON -> extract creator_username, creator_followers, creator_verified.
  7. Parse uploadedAt Unix timestamp -> upload_date, time_trend_days, upload_month.
  8. Extract brand from keyword (OnRunning, Hoka, Saucony, Brooks, etc.).
  9. Add log1p-transformed engagement and log_creator_followers columns.
  10. Save to analysis/merged_dataset.csv and print summary stats.
"""

import ast
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = pathlib.Path("/tmp")
METADATA_PATH = (
    REPO_ROOT
    / "data"
    / "video_metadata"
    / "02_cleaned_metadata"
    / "metadata_final_english.csv"
)
OUTPUT_PATH = ANALYSIS_DIR / "merged_dataset.csv"

# ---------------------------------------------------------------------------
# 1. Load and deduplicate feature CSVs
# ---------------------------------------------------------------------------

def load_feature(path: pathlib.Path, drop_cols: list[str] | None = None) -> pd.DataFrame:
    """Load a feature CSV, cast video_id to str, deduplicate (keep last)."""
    df = pd.read_csv(path, dtype={"video_id": str})
    n_raw = len(df)
    df = df.drop_duplicates(subset="video_id", keep="last")
    n_deduped = len(df)
    if n_raw != n_deduped:
        print(f"  {path.name}: {n_raw} rows -> {n_deduped} after dedup "
              f"({n_raw - n_deduped} duplicates removed)")
    else:
        print(f"  {path.name}: {n_raw} rows (no duplicates)")
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.set_index("video_id")


print("Loading feature CSVs ...")

# Cuts: keep scene_timestamps for potential future use, but it's not numeric
cuts = load_feature(
    FEATURE_DIR / "gcs_cuts.csv",
    drop_cols=["scene_timestamps"],   # complex string, not needed for modeling
)

# Density
density = load_feature(
    FEATURE_DIR / "gcs_density.csv",
    drop_cols=["num_frames_sampled"],
)

# Gaze
gaze = load_feature(
    FEATURE_DIR / "gcs_gaze.csv",
    drop_cols=["num_frames_sampled"],
)

# Object detection: drop object_types (non-numeric string) and num_frames_sampled
obj = load_feature(
    FEATURE_DIR / "gcs_object_detection.csv",
    drop_cols=["object_types", "num_frames_sampled"],
)

# Text detection
text = load_feature(
    FEATURE_DIR / "gcs_text_detection.csv",
    drop_cols=["num_frames_sampled"],
)

# ---------------------------------------------------------------------------
# 2. Load metadata
# ---------------------------------------------------------------------------
print("\nLoading metadata ...")
META_COLS = [
    "id", "views", "likes", "comments", "shares", "bookmarks",
    "uploadedAt", "channel", "keyword", "hashtags",
]
meta = pd.read_csv(METADATA_PATH, dtype={"id": str}, usecols=META_COLS)
meta = meta.rename(columns={"id": "video_id"})
meta = meta.drop_duplicates(subset="video_id", keep="last")
meta = meta.set_index("video_id")
print(f"  metadata: {len(meta)} rows")

# ---------------------------------------------------------------------------
# 3. Inner-join all feature tables
# ---------------------------------------------------------------------------
print("\nMerging feature tables (inner join) ...")
features = cuts.join(density, how="inner") \
               .join(gaze,    how="inner") \
               .join(obj,     how="inner") \
               .join(text,    how="inner")
print(f"  Features after all inner joins: {len(features)} videos, "
      f"{features.shape[1]} columns")

# ---------------------------------------------------------------------------
# 4. Inner-join with metadata
# ---------------------------------------------------------------------------
print("\nJoining with metadata (inner join) ...")
merged = features.join(meta, how="inner")
print(f"  Merged dataset: {merged.shape[0]} videos, {merged.shape[1]} columns")

# Reset index so video_id is a regular column
merged = merged.reset_index()

# ---------------------------------------------------------------------------
# 5. Coerce has_text to int (it's stored as True/False string)
# ---------------------------------------------------------------------------
if merged["has_text"].dtype == object:
    merged["has_text"] = merged["has_text"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    ).astype(float)
else:
    merged["has_text"] = merged["has_text"].astype(float)

# ---------------------------------------------------------------------------
# 6. Parse channel JSON -> creator controls
# ---------------------------------------------------------------------------
print("\nParsing channel JSON ...")

def _parse_channel(raw):
    """Extract (username, followers, verified) from the channel JSON string."""
    if pd.isna(raw):
        return None, np.nan, np.nan
    try:
        d = ast.literal_eval(str(raw))
        return d.get("username"), d.get("followers", np.nan), int(bool(d.get("verified", False)))
    except Exception:
        return None, np.nan, np.nan

parsed = merged["channel"].apply(_parse_channel)
merged["creator_username"] = [x[0] for x in parsed]
merged["creator_followers"] = pd.to_numeric([x[1] for x in parsed], errors="coerce")
merged["creator_verified"]  = [x[2] for x in parsed]
merged["log_creator_followers"] = np.log1p(merged["creator_followers"].clip(lower=0))
print(f"  creator_followers: {merged['creator_followers'].notna().sum():,} non-null, "
      f"median={merged['creator_followers'].median():,.0f}")

# ---------------------------------------------------------------------------
# 7. Parse uploadedAt -> time controls
# ---------------------------------------------------------------------------
print("\nParsing upload timestamps ...")
merged["upload_date"] = pd.to_datetime(merged["uploadedAt"], unit="s", utc=True)
start_date = merged["upload_date"].min()
merged["time_trend_days"] = (merged["upload_date"] - start_date).dt.days
merged["upload_month"]    = merged["upload_date"].dt.month.astype(str).str.zfill(2)
merged["upload_year"]     = merged["upload_date"].dt.year
print(f"  Date range: {start_date.date()} to {merged['upload_date'].max().date()}")
print(f"  time_trend_days: 0 – {merged['time_trend_days'].max():.0f}")

# ---------------------------------------------------------------------------
# 8. Extract brand from keyword
# ---------------------------------------------------------------------------
BRAND_PATTERNS = {
    "OnRunning":  ["onrunning", "on running", "on cloud"],
    "Hoka":       ["hoka"],
    "Saucony":    ["saucony"],
    "Brooks":     ["brooks"],
    "Nike":       ["nike"],
    "Adidas":     ["adidas"],
    "ASICS":      ["asics"],
    "NewBalance": ["new balance"],
    "Salomon":    ["salomon"],
    "Altra":      ["altra"],
}

def _extract_brand(kw):
    if pd.isna(kw):
        return "Other"
    kw_lower = kw.lower()
    for brand, patterns in BRAND_PATTERNS.items():
        if any(p in kw_lower for p in patterns):
            return brand
    return "Other"

merged["brand"] = merged["keyword"].apply(_extract_brand)
print(f"\nBrand distribution:\n{merged['brand'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# 9. Add log1p engagement columns
# ---------------------------------------------------------------------------
ENGAGEMENT_COLS = ["views", "likes", "comments", "shares", "bookmarks"]
for col in ENGAGEMENT_COLS:
    merged[f"log_{col}"] = np.log1p(merged[col].clip(lower=0))

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Rows    : {len(merged):,}")
print(f"Columns : {merged.shape[1]}")

print("\n--- Column list ---")
for col in merged.columns:
    print(f"  {col}")

print("\n--- Missing values (columns with any NaN) ---")
missing = merged.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  None")
else:
    print(missing.to_string())

print("\n--- Engagement stats (raw) ---")
print(merged[ENGAGEMENT_COLS].describe().round(1).to_string())

print("\n--- Engagement stats (log1p) ---")
log_cols = [f"log_{c}" for c in ENGAGEMENT_COLS]
print(merged[log_cols].describe().round(3).to_string())

print("\n--- Feature stats ---")
feature_cols = [
    "cut_count", "cuts_per_second", "avg_scene_duration",
    "min_scene_duration", "max_scene_duration", "duration_sec", "fps",
    "color_entropy", "edge_density_ratio", "avg_motion_magnitude",
    "num_faces", "gaze_at_camera_ratio", "avg_gaze_yaw", "avg_gaze_pitch",
    "avg_objects", "num_humans",
    "has_text", "text_area_ratio", "avg_text_regions", "text_changes_per_second",
]
feature_cols = [c for c in feature_cols if c in merged.columns]
print(merged[feature_cols].describe().round(3).to_string())
