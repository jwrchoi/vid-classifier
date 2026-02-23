"""
Generate an inter-coder reliability (ICR) queue from a reference coder's annotations.

Creates gs://vid-classifier-db/annotations/queue.csv containing only the frames
the reference coder has annotated. Other coders will then see only those frames.

Usage:
    python annotation_dashboard/scripts/generate_icr_queue.py [--coder Soojin]

To revert to full 500-frame queue:
    gsutil rm gs://vid-classifier-db/annotations/queue.csv
"""

import argparse
import io
import pandas as pd
from google.cloud import storage

BUCKET = "vid-classifier-db"
ANNOTATIONS_PATH = "annotations/annotations.csv"
QUEUE_PATH = "annotations/queue.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coder", default="Soojin", help="Reference coder name")
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # Read annotations
    blob = bucket.blob(ANNOTATIONS_PATH)
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    df["video_id"] = df["video_id"].astype(str)

    # Filter to reference coder's annotated frames
    coder_df = df[df["annotator"].str.lower() == args.coder.lower()].copy()
    coder_df = coder_df.dropna(subset=["frame_index"])
    coder_df["frame_index"] = coder_df["frame_index"].astype(int)
    pairs = coder_df[["video_id", "frame_index"]].drop_duplicates()

    print(f"Found {len(pairs)} unique (video_id, frame_index) pairs for '{args.coder}'")

    # Write queue.csv
    queue_csv = pairs.to_csv(index=False)
    bucket.blob(QUEUE_PATH).upload_from_string(queue_csv, content_type="text/csv")
    print(f"Uploaded ICR queue to gs://{BUCKET}/{QUEUE_PATH}")
    print("Other coders will now see only these frames.")
    print()
    print("To revert to full queue:")
    print(f"    gsutil rm gs://{BUCKET}/{QUEUE_PATH}")


if __name__ == "__main__":
    main()
