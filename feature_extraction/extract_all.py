"""
Feature Extraction Orchestrator
================================

Downloads videos from GCS, runs one or more feature extractors, and saves
the results as CSV files in data/features/.

Usage:
    # Run all extractors on all videos:
    python extract_all.py

    # Run only cut detection on the first 10 videos:
    python extract_all.py --extractors cuts --limit 10

    # Specify a local video list CSV instead of scanning GCS:
    python extract_all.py --video-list ../data/video_list_v2.csv

    # Upload results to GCS every 25 videos (for spot VM resilience):
    python extract_all.py --checkpoint-interval 25
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the monorepo root is on sys.path so `shared` and
# `feature_extraction` packages are importable.
# ---------------------------------------------------------------------------
MONOREPO_ROOT = Path(__file__).resolve().parent.parent
if str(MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MONOREPO_ROOT))

from shared.gcs_utils import download_video_to_temp, list_video_blobs, upload_file_to_gcs  # noqa: E402
from shared.video_utils import extract_video_id  # noqa: E402
from shared.config import GCS_BUCKET_NAME, GCS_VIDEO_PREFIX, FEATURES_DIR  # noqa: E402

# Minimum free disk space (in bytes) required before downloading a video.
# If free space drops below this, the pipeline pauses and warns.
MIN_FREE_DISK_BYTES = 1_000_000_000  # 1 GB

# ---------------------------------------------------------------------------
# Registry of available extractors.
# Each key maps to an (module_path, function_name) pair.
# The function signature is:  extract(video_path: Path, video_id: str) -> dict
# ---------------------------------------------------------------------------
EXTRACTOR_REGISTRY = {
    "cuts":             ("feature_extraction.extractors.cut_detection",    "extract"),
    "density":          ("feature_extraction.extractors.density",          "extract"),
    "gaze":             ("feature_extraction.extractors.gaze",             "extract"),
    "object_detection": ("feature_extraction.extractors.object_detection", "extract"),
    "text_detection":   ("feature_extraction.extractors.text_detection",   "extract"),
}


def load_extractor(name: str):
    """
    Dynamically import and return the extract function for a named extractor.

    Args:
        name: Key in EXTRACTOR_REGISTRY (e.g. "cuts").

    Returns:
        The extract() callable.
    """
    module_path, func_name = EXTRACTOR_REGISTRY[name]
    # __import__ with fromlist forces the submodule to be returned directly.
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def get_video_list_from_csv(csv_path: Path) -> list[dict]:
    """
    Read a video list CSV (must have a 'blob_name' or 'video_id' column).

    Returns:
        List of dicts with 'blob_name' and 'video_id' keys.
    """
    df = pd.read_csv(csv_path)

    # video_list_v2.csv has a 'gcs_path' column with the full blob path.
    if "gcs_path" in df.columns:
        records = []
        for _, row in df.iterrows():
            blob = row["gcs_path"]
            vid = str(row.get("video_id", extract_video_id(blob)))
            records.append({"blob_name": blob, "video_id": vid})
        return records

    # The annotation dashboard's video list has a 'blob_name' column.
    if "blob_name" in df.columns:
        records = []
        for _, row in df.iterrows():
            blob = row["blob_name"]
            vid = str(row.get("video_id", extract_video_id(blob)))
            records.append({"blob_name": blob, "video_id": vid})
        return records

    # Fallback: assume 'video_id' column; reconstruct blob path.
    if "video_id" in df.columns:
        return [
            {
                "blob_name": f"{GCS_VIDEO_PREFIX}{row['video_id']}.mp4",
                "video_id": str(row["video_id"]),
            }
            for _, row in df.iterrows()
        ]

    raise ValueError(f"CSV {csv_path} must have 'gcs_path', 'blob_name', or 'video_id' column")


def get_video_list_from_gcs() -> list[dict]:
    """
    Scan GCS for all videos and return a list of dicts.
    """
    blobs = list_video_blobs(GCS_BUCKET_NAME, GCS_VIDEO_PREFIX)
    return [
        {"blob_name": b, "video_id": extract_video_id(b)}
        for b in blobs
    ]


def _check_disk_space(path: Path = Path("/tmp")) -> int:
    """Return free disk space in bytes at the given path."""
    usage = shutil.disk_usage(str(path))
    return usage.free


def _upload_checkpoint(extractor_names: list[str], gcs_prefix: str = "features") -> None:
    """Upload current CSV files to GCS as a checkpoint."""
    for name in extractor_names:
        csv_path = FEATURES_DIR / f"{name}.csv"
        if csv_path.exists():
            dest = f"{gcs_prefix}/{name}.csv"
            try:
                upload_file_to_gcs(csv_path, dest)
            except Exception as e:
                print(f"  [WARN] Checkpoint upload failed for {name}: {e}")


def run_extraction(
    extractor_names: list[str],
    video_list: list[dict],
    limit: int | None = None,
    skip_existing: bool = True,
    checkpoint_interval: int = 0,
) -> None:
    """
    Main extraction loop.

    For each video, downloads it from GCS to a temp file, runs each
    requested extractor, and appends results to per-extractor CSVs.

    Args:
        extractor_names:     List of extractor keys to run (e.g. ["cuts"]).
        video_list:          List of dicts with 'blob_name' and 'video_id'.
        limit:               Max videos to process (None = all).
        skip_existing:       If True, skip videos already in the output CSV.
        checkpoint_interval: Upload CSVs to GCS every N videos (0 = disabled).
                             Useful on spot/preemptible VMs to survive preemption.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load extractor functions.
    extractors = {}
    for name in extractor_names:
        if name not in EXTRACTOR_REGISTRY:
            print(f"[WARN] Unknown extractor '{name}', skipping.")
            continue
        extractors[name] = load_extractor(name)

    if not extractors:
        print("[ERROR] No valid extractors specified.")
        return

    # Load existing results so we can skip already-processed videos.
    existing_ids: dict[str, set] = {}
    for name in extractors:
        csv_path = FEATURES_DIR / f"{name}.csv"
        if csv_path.exists() and skip_existing:
            df = pd.read_csv(csv_path)
            existing_ids[name] = set(df["video_id"].astype(str))
        else:
            existing_ids[name] = set()

    # Apply limit.
    videos = video_list[:limit] if limit else video_list
    total = len(videos)

    # Count how many are already done (for accurate progress tracking).
    already_done = sum(
        1 for v in videos
        if all(v["video_id"] in existing_ids[n] for n in extractors)
    )

    print(f"\n{'='*60}")
    print(f"Feature Extraction Pipeline")
    print(f"  Extractors : {', '.join(extractors.keys())}")
    print(f"  Videos     : {total} ({already_done} already processed)")
    print(f"  Output dir : {FEATURES_DIR}")
    if checkpoint_interval > 0:
        print(f"  Checkpoint : every {checkpoint_interval} videos → GCS")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0
    videos_since_checkpoint = 0
    pipeline_start = time.time()

    for i, video in enumerate(videos, 1):
        blob_name = video["blob_name"]
        video_id = video["video_id"]

        # Check if all extractors already have this video.
        all_done = all(video_id in existing_ids[n] for n in extractors)
        if skip_existing and all_done:
            continue

        # --- Disk space guard ---
        free_bytes = _check_disk_space()
        if free_bytes < MIN_FREE_DISK_BYTES:
            free_gb = free_bytes / (1024 ** 3)
            print(f"\n[ERROR] Low disk space ({free_gb:.1f} GB free). "
                  f"Stopping to prevent data corruption.")
            print(f"  Processed {success_count} videos successfully before stopping.")
            # Upload what we have before bailing out.
            if checkpoint_interval > 0:
                print("  Uploading checkpoint to GCS...")
                _upload_checkpoint(list(extractors.keys()))
            return

        # --- Progress with ETA ---
        processed_so_far = success_count + fail_count
        remaining = (total - already_done) - processed_so_far
        elapsed_total = time.time() - pipeline_start
        if processed_so_far > 0:
            avg_per_video = elapsed_total / processed_so_far
            eta_sec = avg_per_video * remaining
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        else:
            eta_str = "estimating..."

        print(f"[{i}/{total}] {video_id} (remaining: {remaining}, "
              f"ETA: {eta_str}) — downloading...", end=" ", flush=True)
        t0 = time.time()

        tmp_path = None
        try:
            tmp_path = download_video_to_temp(blob_name)
        except Exception as e:
            print(f"DOWNLOAD FAILED: {e}")
            fail_count += 1
            continue

        try:
            for name, extract_fn in extractors.items():
                if skip_existing and video_id in existing_ids[name]:
                    continue

                # Run the extractor.
                features = extract_fn(tmp_path, video_id)

                # Append to CSV (immediate flush for checkpoint safety).
                csv_path = FEATURES_DIR / f"{name}.csv"
                df_row = pd.DataFrame([features])
                if csv_path.exists():
                    df_row.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df_row.to_csv(csv_path, index=False)

                existing_ids[name].add(video_id)

            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            success_count += 1
            videos_since_checkpoint += 1

        except Exception as e:
            print(f"EXTRACTION FAILED: {e}")
            fail_count += 1

        finally:
            # Always clean up the temp file to reclaim disk space.
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # --- Periodic GCS checkpoint ---
        if (checkpoint_interval > 0
                and videos_since_checkpoint >= checkpoint_interval):
            print(f"  >> Checkpointing to GCS ({success_count} videos done)...")
            _upload_checkpoint(list(extractors.keys()))
            videos_since_checkpoint = 0

    # Final summary and checkpoint.
    elapsed_total = time.time() - pipeline_start
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
    print(f"\n{'='*60}")
    print(f"Extraction complete in {elapsed_str}")
    print(f"  Success: {success_count}  |  Failed: {fail_count}  |  Skipped: {already_done}")
    print(f"  Results: {FEATURES_DIR}/")

    if checkpoint_interval > 0 and success_count > 0:
        print(f"  Uploading final results to GCS...")
        _upload_checkpoint(list(extractors.keys()))
        print(f"  Uploaded to gs://{GCS_BUCKET_NAME}/features/")

    print(f"{'='*60}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run feature extraction on TikTok videos from GCS."
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=list(EXTRACTOR_REGISTRY.keys()),
        help=f"Which extractors to run. Choices: {list(EXTRACTOR_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of videos to process.",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        default=None,
        help="Path to a video list CSV. If omitted, scans GCS.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract features even if video already in output CSV.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Upload CSVs to GCS every N videos (0 = disabled). "
             "Recommended on spot/preemptible VMs for resilience.",
    )
    args = parser.parse_args()

    # Build video list.
    if args.video_list:
        videos = get_video_list_from_csv(args.video_list)
    else:
        print("Scanning GCS for videos...")
        videos = get_video_list_from_gcs()

    run_extraction(
        extractor_names=args.extractors,
        video_list=videos,
        limit=args.limit,
        skip_existing=not args.no_skip,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
