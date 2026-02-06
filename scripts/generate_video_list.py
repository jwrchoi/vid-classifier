#!/usr/bin/env python3
"""
Generate a fixed video list for reliability testing.

Connects to GCS, lists all videos in the bucket, randomly samples 50,
and writes data/video_list_v1.csv. This CSV is committed to the repo
so all coders annotate the same videos in the same order.

Usage:
    python scripts/generate_video_list.py
"""

import sys
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GCS_BUCKET_NAME, GCS_VIDEO_PREFIX, VIDEO_LIST_FILE, ensure_output_dir
from utils.gcs import list_video_blobs
from utils.video_processing import extract_video_id

SAMPLE_SIZE = 50
RANDOM_SEED = 2026


def main():
    print(f"Listing videos in gs://{GCS_BUCKET_NAME}/{GCS_VIDEO_PREFIX} ...")
    blobs = list_video_blobs(GCS_BUCKET_NAME, GCS_VIDEO_PREFIX)

    # Filter out macOS resource fork files (._*) and keep only real videos
    blobs = [b for b in blobs if not Path(b).name.startswith('._')]
    print(f"Found {len(blobs)} videos (after filtering).")

    if len(blobs) < SAMPLE_SIZE:
        print(f"Warning: only {len(blobs)} videos available, using all of them.")
        sample = blobs
    else:
        random.seed(RANDOM_SEED)
        sample = random.sample(blobs, SAMPLE_SIZE)

    # Sort for deterministic order after sampling
    sample.sort()

    ensure_output_dir()

    with open(VIDEO_LIST_FILE, 'w') as f:
        f.write("order,video_id,filename,gcs_path\n")
        for i, blob_name in enumerate(sample, start=1):
            filename = Path(blob_name).name
            video_id = extract_video_id(filename)
            f.write(f"{i},{video_id},{filename},{blob_name}\n")

    print(f"Wrote {len(sample)} videos to {VIDEO_LIST_FILE}")


if __name__ == "__main__":
    main()
