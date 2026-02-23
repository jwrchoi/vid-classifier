"""
Generate ICR v2 per-coder queues for the second inter-coder reliability round.

Queue contents
--------------
  queue_Soojin.csv  —  Soojin's blank-annotation frames (re-codes) + 30 new ICR frames
  queue_Janice.csv  —  30 new ICR frames only

The 30 new frames are the same for both coders (shared ICR set), but presented in
a different shuffled order per coder.  After both coders complete their queues,
run compute_icr.py to get reliability numbers on the fresh 30-frame set.

Steps performed
---------------
  1. Back up annotations.csv on GCS
  2. Find coder1's frames that need re-coding (blank perspective or distance,
     no_human_visible=False)
  3. Sample N new frames not yet coded by either coder (fixed seed → reproducible)
  4. Upload queue_Coder1.csv and queue_Coder2.csv to GCS
  5. Delete the old global queue.csv so it no longer overrides other coders

Usage
-----
    python annotation_dashboard/scripts/generate_icr_v2_queue.py \\
        [--coder1 Soojin] [--coder2 Janice] [--new-frames 30] [--seed 42]

To revert (restores full shuffled queue for all coders):
    gsutil rm gs://vid-classifier-db/annotations/queue_Soojin.csv
    gsutil rm gs://vid-classifier-db/annotations/queue_Janice.csv
"""

import argparse
import hashlib
import io
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from google.cloud import storage

BUCKET = "vid-classifier-db"
ANNOTATIONS_GCS_PATH = "annotations/annotations.csv"
GLOBAL_QUEUE_GCS_PATH = "annotations/queue.csv"

# Local video list (used to enumerate all possible frames)
VIDEO_LIST_FILE = Path(__file__).parents[2] / "data" / "video_list_v2.csv"
FRAMES_PER_VIDEO = 10


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def backup_annotations(bucket, timestamp: str) -> None:
    dst = f"annotations/backups/annotations_backup_{timestamp}.csv"
    data = bucket.blob(ANNOTATIONS_GCS_PATH).download_as_bytes()
    bucket.blob(dst).upload_from_string(data, content_type="text/csv")
    print(f"Backup → gs://{BUCKET}/{dst}")


def fetch_annotations(bucket) -> pd.DataFrame:
    data = bucket.blob(ANNOTATIONS_GCS_PATH).download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    df["video_id"] = df["video_id"].astype(str)
    df["no_human_visible"] = df["no_human_visible"].map(
        lambda v: str(v).strip().lower() == "true"
    )
    return df


def upload_queue(bucket, pairs: list[tuple[str, int]], gcs_path: str) -> None:
    df = pd.DataFrame(pairs, columns=["video_id", "frame_index"])
    bucket.blob(gcs_path).upload_from_string(
        df.to_csv(index=False).encode(), content_type="text/csv"
    )
    print(f"Uploaded {len(pairs)}-frame queue → gs://{BUCKET}/{gcs_path}")


def delete_global_queue(bucket) -> None:
    blob = bucket.blob(GLOBAL_QUEUE_GCS_PATH)
    if blob.exists():
        blob.delete()
        print(f"Deleted global queue  → gs://{BUCKET}/{GLOBAL_QUEUE_GCS_PATH}")
    else:
        print(f"Global queue not found (already removed): gs://{BUCKET}/{GLOBAL_QUEUE_GCS_PATH}")


# ---------------------------------------------------------------------------
# Frame selection logic
# ---------------------------------------------------------------------------

def get_blank_frames(df: pd.DataFrame, coder: str) -> list[tuple[str, int]]:
    """
    Return (video_id, frame_index) pairs where coder saved a row but
    perspective or distance is blank and no_human_visible is False.
    """
    sub = df[df["annotator"].str.lower() == coder.lower()].copy()
    sub = sub.dropna(subset=["frame_index"])
    sub["frame_index"] = sub["frame_index"].astype(int)

    persp_blank = sub["perspective"].isna() | (sub["perspective"].fillna("").str.strip() == "")
    dist_blank  = sub["distance"].isna()    | (sub["distance"].fillna("").str.strip() == "")
    mask = ~sub["no_human_visible"] & (persp_blank | dist_blank)

    pairs = sub[mask][["video_id", "frame_index"]].drop_duplicates()
    return list(zip(pairs["video_id"], pairs["frame_index"].astype(int)))


def get_all_coded_pairs(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Return every (video_id, frame_index) coded by any coder."""
    sub = df.dropna(subset=["frame_index"]).copy()
    sub["frame_index"] = sub["frame_index"].astype(int)
    return set(zip(sub["video_id"], sub["frame_index"]))


def sample_new_frames(
    coded_pairs: set[tuple[str, int]],
    n: int,
    seed: int,
) -> list[tuple[str, int]]:
    """
    Sample n (video_id, frame_index) pairs not yet coded by anyone.
    Draws from the full video_list_v2.csv × FRAMES_PER_VIDEO pool.
    """
    df_vids = pd.read_csv(VIDEO_LIST_FILE)
    df_vids["video_id"] = df_vids["video_id"].astype(str)

    pool = [
        (row["video_id"], fi)
        for _, row in df_vids.iterrows()
        for fi in range(FRAMES_PER_VIDEO)
        if (row["video_id"], fi) not in coded_pairs
    ]

    if len(pool) < n:
        raise ValueError(
            f"Only {len(pool)} uncoded frames available in the video list; "
            f"requested {n}. Lower --new-frames or add more videos."
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def shuffled_for_coder(
    pairs: list[tuple[str, int]], coder: str, seed_salt: str = "icr_v2"
) -> list[tuple[str, int]]:
    """Deterministically shuffle pairs for this coder (reproducible per-coder order)."""
    seed = int(
        hashlib.md5((coder.title() + seed_salt).encode()).hexdigest(), 16
    ) % (2**32)
    rng = random.Random(seed)
    result = list(pairs)
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coder1", default="Soojin",
                        help="Coder whose blank frames need re-coding (default: Soojin)")
    parser.add_argument("--coder2", default="Janice",
                        help="Second coder (default: Janice)")
    parser.add_argument("--new-frames", type=int, default=30,
                        help="Number of new shared ICR frames to sample (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for new frame sampling (default: 42)")
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Backup
    print("=== Step 1: Backup ===")
    backup_annotations(bucket, timestamp)

    # 2. Read annotations
    print("\n=== Step 2: Read annotations ===")
    df = fetch_annotations(bucket)
    print(f"  Total rows : {len(df)}")
    print(f"  Coders     : {df['annotator'].value_counts().to_dict()}")

    # 3. Find coder1's blank frames
    print(f"\n=== Step 3: {args.coder1}'s blank-annotation frames ===")
    blank_pairs = get_blank_frames(df, args.coder1)
    print(f"  Found {len(blank_pairs)} frame(s) needing re-code:")
    for vid, fi in blank_pairs:
        row = df[
            (df["annotator"].str.lower() == args.coder1.lower())
            & (df["video_id"] == vid)
            & (df["frame_index"] == fi)
        ].iloc[-1]
        persp = row.get("perspective", "")
        dist  = row.get("distance", "")
        print(f"    video={vid}  frame={fi}  perspective={persp!r}  distance={dist!r}")

    # 4. Sample new frames
    print(f"\n=== Step 4: Sample {args.new_frames} new frames (seed={args.seed}) ===")
    coded_pairs = get_all_coded_pairs(df)
    print(f"  Already-coded pairs (all coders): {len(coded_pairs)}")
    new_pairs = sample_new_frames(coded_pairs, args.new_frames, args.seed)
    print(f"  Sampled {len(new_pairs)} new frames")

    # 5. Build per-coder queues
    print("\n=== Step 5: Build per-coder queues ===")
    # Coder1: blank re-dos + new frames, shuffled for coder1
    coder1_pairs = shuffled_for_coder(blank_pairs + new_pairs, args.coder1)
    # Coder2: new frames only, shuffled for coder2
    coder2_pairs = shuffled_for_coder(new_pairs, args.coder2)

    print(f"  {args.coder1}: {len(coder1_pairs)} frames "
          f"({len(blank_pairs)} re-codes + {len(new_pairs)} new)")
    print(f"  {args.coder2}: {len(coder2_pairs)} frames ({len(new_pairs)} new)")

    # 6. Upload per-coder queues
    print("\n=== Step 6: Upload per-coder queues ===")
    upload_queue(bucket, coder1_pairs,
                 f"annotations/queue_{args.coder1.title()}.csv")
    upload_queue(bucket, coder2_pairs,
                 f"annotations/queue_{args.coder2.title()}.csv")

    # 7. Delete global queue.csv
    print("\n=== Step 7: Delete global queue.csv ===")
    delete_global_queue(bucket)

    print("\n=== Done ===")
    print(f"  {args.coder1}: reload dashboard → sees {len(coder1_pairs)} frames "
          f"({len(blank_pairs)} re-codes first, then {len(new_pairs)} new)")
    print(f"  {args.coder2}: reload dashboard → sees {len(new_pairs)} new frames")
    print("\nAfter both coders finish, re-run compute_icr.py to get updated ICR numbers.")
    print("\nTo revert (restores full shuffled queue for all coders):")
    print(f"  gsutil rm gs://{BUCKET}/annotations/queue_{args.coder1.title()}.csv")
    print(f"  gsutil rm gs://{BUCKET}/annotations/queue_{args.coder2.title()}.csv")


if __name__ == "__main__":
    main()
