#!/usr/bin/env python3
"""
Copy filtered videos to a new GCS prefix and build the full video list.

Reads metadata_final_english.csv (output from the notebook), cross-references
with blobs in gs://vid-classifier-db/videos/00_videos/, copies matching videos
to gs://vid-classifier-db/videos/01_filtered/, and writes data/video_list_v2.csv.

Usage:
    python scripts/build_filtered_video_list.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import google.auth
from google.cloud import storage

from config import GCS_BUCKET_NAME, OUTPUT_DIR, ensure_output_dir
from utils.video_processing import extract_video_id

# Paths
METADATA_CSV = Path("data/video_metadata/02_cleaned_metadata/metadata_final_english.csv")
SRC_PREFIX = "videos/00_videos/"
DST_PREFIX = "videos/01_filtered/"
OUTPUT_CSV = OUTPUT_DIR / "video_list_v2.csv"


def get_gcs_client():
    """Return an authenticated GCS client (no quota project)."""
    credentials, project = google.auth.default()
    credentials = credentials.with_quota_project(None)
    return storage.Client(project=project, credentials=credentials)


def main():
    # 1. Load filtered metadata
    print(f"Loading metadata from {METADATA_CSV} ...")
    df = pd.read_csv(METADATA_CSV, dtype={"id": "string"})
    valid_ids = set(df["id"].dropna())
    print(f"  Filtered metadata IDs: {len(valid_ids)}")

    # 2. List source blobs in GCS
    print(f"\nListing blobs in gs://{GCS_BUCKET_NAME}/{SRC_PREFIX} ...")
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    src_blobs = list(bucket.list_blobs(prefix=SRC_PREFIX))

    # Filter to video files, skip macOS resource forks
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    src_blobs = [
        b for b in src_blobs
        if any(b.name.lower().endswith(ext) for ext in video_exts)
        and not Path(b.name).name.startswith("._")
    ]
    print(f"  Total video blobs in source: {len(src_blobs)}")

    # 3. Build lookup: video_id -> blob
    blob_by_id = {}
    for blob in src_blobs:
        filename = Path(blob.name).name
        vid_id = extract_video_id(filename)
        if vid_id:
            blob_by_id[vid_id] = blob

    # 4. Match metadata IDs to GCS blobs
    matched_ids = valid_ids & set(blob_by_id.keys())
    print(f"  Matched to metadata: {len(matched_ids)}")

    # 5. List existing blobs in destination (for idempotency)
    print(f"\nChecking existing blobs in gs://{GCS_BUCKET_NAME}/{DST_PREFIX} ...")
    existing_dst = set()
    for blob in bucket.list_blobs(prefix=DST_PREFIX):
        existing_dst.add(Path(blob.name).name)
    print(f"  Already in destination: {len(existing_dst)}")

    # 6. Copy matched blobs to destination
    matched_list = sorted(matched_ids)
    copied = 0
    skipped = 0

    print(f"\nCopying {len(matched_list)} videos to {DST_PREFIX} ...")
    for i, vid_id in enumerate(matched_list, start=1):
        src_blob = blob_by_id[vid_id]
        filename = Path(src_blob.name).name
        dst_blob_name = DST_PREFIX + filename

        if filename in existing_dst:
            skipped += 1
        else:
            dst_blob = bucket.blob(dst_blob_name)
            # Server-side copy (no download/upload)
            rewrite_token = None
            while True:
                rewrite_token, bytes_rewritten, total_bytes = dst_blob.rewrite(
                    src_blob, token=rewrite_token
                )
                if rewrite_token is None:
                    break
            copied += 1

        if i % 500 == 0 or i == len(matched_list):
            print(f"  Progress: {i}/{len(matched_list)} (copied: {copied}, skipped: {skipped})")

    print(f"\nCopy complete: {copied} copied, {skipped} already existed")

    # 7. Write video_list_v2.csv (full list, not sampled)
    ensure_output_dir()
    rows = []
    for vid_id in matched_list:
        src_blob = blob_by_id[vid_id]
        filename = Path(src_blob.name).name
        gcs_path = DST_PREFIX + filename
        rows.append((vid_id, filename, gcs_path))

    with open(OUTPUT_CSV, "w") as f:
        f.write("order,video_id,filename,gcs_path\n")
        for i, (vid_id, filename, gcs_path) in enumerate(rows, start=1):
            f.write(f"{i},{vid_id},{filename},{gcs_path}\n")

    print(f"\nWrote {len(rows)} videos to {OUTPUT_CSV}")
    print(f"\nSummary:")
    print(f"  Total in GCS source:   {len(src_blobs)}")
    print(f"  Matched to metadata:   {len(matched_ids)}")
    print(f"  Copied to filtered:    {copied}")
    print(f"  Video list entries:    {len(rows)}")


if __name__ == "__main__":
    main()
