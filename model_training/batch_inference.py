"""
Parallel batch inference on ~15K TikTok videos using ResNet-50 models.

Architecture:
    [GCS video list] → [N download workers (processes)] → [Queue] → [GPU inference (main)]

Produces frame-level and video-level prediction CSVs.

Usage:
    python -m model_training.batch_inference \
        --pov-weights path/to/pov.pth \
        --dist-weights path/to/dist.pth \
        --workers 4 --batch-size 64 --checkpoint-interval 500

    # Quick local test:
    python -m model_training.batch_inference --limit 5 \
        --pov-weights path/to/pov.pth --dist-weights path/to/dist.pth
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training.config import (
    FRAMES_PER_VIDEO,
    GCS_BUCKET_NAME,
    IMG_SIZE,
    MODEL_CONFIGS,
    find_models_dir,
)
from model_training.dataset import get_eval_transform
from model_training.trainer import build_model, get_device
from shared.video_utils import extract_video_id

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = _REPO_ROOT / "data" / "features"
FRAME_CSV = FEATURES_DIR / "model_predictions.csv"
VIDEO_CSV = FEATURES_DIR / "model_predictions_video.csv"

FRAME_HEADER = [
    "video_id", "frame_index",
    "pred_perspective", "perspective_conf", "perspective_probs",
    "pred_distance", "distance_conf", "distance_probs",
]
VIDEO_HEADER = [
    "video_id", "pred_perspective", "perspective_conf",
    "pred_distance", "distance_conf", "n_frames",
]

# Sentinel to signal workers are done
_SENTINEL = None


# ---------------------------------------------------------------------------
# Producer: download video + extract frames
# ---------------------------------------------------------------------------

def _extract_frames_from_video(video_path: str, num_frames: int) -> List[np.ndarray]:
    """Extract evenly-spaced frames as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    if num_frames >= total:
        indices = list(range(total))
    else:
        step = total / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _producer_worker(blob_name: str) -> Optional[Tuple[str, List[np.ndarray]]]:
    """Download a single video from GCS and extract frames.

    Runs in a worker process. Returns (video_id, frames) or None on failure.
    """
    try:
        from shared.gcs_utils import get_gcs_client

        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        suffix = Path(blob_name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            blob.download_to_filename(tmp.name)
            tmp.close()
            frames = _extract_frames_from_video(tmp.name, FRAMES_PER_VIDEO)
        finally:
            os.unlink(tmp.name)

        if not frames:
            return None

        video_id = extract_video_id(Path(blob_name).name)
        return (video_id, frames)

    except Exception as e:
        print(f"  [producer] Error on {blob_name}: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Consumer: GPU inference
# ---------------------------------------------------------------------------

def run_inference(
    pov_weights: str,
    dist_weights: str,
    blob_names: List[str],
    workers: int = 4,
    batch_size: int = 64,
    checkpoint_interval: int = 500,
):
    """Main inference loop: producers download, consumer runs GPU models."""
    device = get_device()
    print(f"Device: {device}")
    print(f"Videos to process: {len(blob_names)}")
    print(f"Workers: {workers}, Batch size: {batch_size}")

    # Load models
    print("Loading POV model...")
    pov_cfg = MODEL_CONFIGS["perspective"]
    pov_model = build_model(pov_cfg["num_classes"], pov_weights, device)
    pov_model.eval()

    print("Loading distance model...")
    dist_cfg = MODEL_CONFIGS["distance"]
    dist_model = build_model(dist_cfg["num_classes"], dist_weights, device)
    dist_model.eval()

    transform = get_eval_transform()

    # Prepare output
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint to resume
    completed_ids = set()
    if FRAME_CSV.exists():
        existing = pd.read_csv(FRAME_CSV)
        existing["video_id"] = existing["video_id"].astype(str)
        completed_ids = set(existing["video_id"].unique())
        print(f"Resuming — {len(completed_ids)} videos already done")

    remaining_blobs = [
        b for b in blob_names
        if extract_video_id(Path(b).name) not in completed_ids
    ]
    print(f"Remaining: {len(remaining_blobs)} videos")

    if not remaining_blobs:
        print("All videos already processed.")
        _aggregate_video_level()
        return

    # Open CSV for appending
    write_header = not FRAME_CSV.exists()
    frame_file = open(FRAME_CSV, "a", newline="")
    frame_writer = csv.writer(frame_file)
    if write_header:
        frame_writer.writerow(FRAME_HEADER)

    # Process with multiprocessing pool
    processed = 0
    start_time = time.time()

    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_producer_worker, remaining_blobs):
            if result is None:
                processed += 1
                continue

            video_id, frames = result

            # Transform frames to tensor batch
            tensors = []
            for frame in frames:
                img = Image.fromarray(frame)
                tensors.append(transform(img))
            batch = torch.stack(tensors).to(device)

            # Run both models
            with torch.no_grad():
                pov_logits = pov_model(batch)
                dist_logits = dist_model(batch)

            pov_probs = torch.softmax(pov_logits, dim=1).cpu().numpy()
            dist_probs = torch.softmax(dist_logits, dim=1).cpu().numpy()

            pov_preds = pov_logits.argmax(dim=1).cpu().numpy()
            dist_preds = dist_logits.argmax(dim=1).cpu().numpy()

            # Write frame-level rows
            for fi in range(len(frames)):
                pov_class = pov_cfg["classes"][pov_preds[fi]]
                pov_conf = float(pov_probs[fi][pov_preds[fi]])
                pov_probs_str = ";".join(f"{p:.4f}" for p in pov_probs[fi])

                dist_class = dist_cfg["classes"][dist_preds[fi]]
                dist_conf = float(dist_probs[fi][dist_preds[fi]])
                dist_probs_str = ";".join(f"{p:.4f}" for p in dist_probs[fi])

                frame_writer.writerow([
                    video_id, fi,
                    pov_class, f"{pov_conf:.4f}", pov_probs_str,
                    dist_class, f"{dist_conf:.4f}", dist_probs_str,
                ])

            processed += 1

            # Checkpoint: flush + upload to GCS
            if processed % checkpoint_interval == 0:
                frame_file.flush()
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (len(remaining_blobs) - processed) / rate if rate > 0 else 0
                print(
                    f"  Checkpoint: {processed}/{len(remaining_blobs)} "
                    f"({rate:.1f} vid/s, ETA {eta/60:.0f}min)",
                    flush=True,
                )
                _upload_checkpoint()

            elif processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                print(
                    f"  Progress: {processed}/{len(remaining_blobs)} ({rate:.1f} vid/s)",
                    flush=True,
                )

    frame_file.close()

    # Final upload
    elapsed = time.time() - start_time
    print(f"\nProcessed {processed} videos in {elapsed/60:.1f} min")
    _upload_checkpoint()

    # Aggregate video-level predictions
    _aggregate_video_level()


def _upload_checkpoint():
    """Upload frame CSV to GCS."""
    try:
        from shared.gcs_utils import upload_file_to_gcs
        if FRAME_CSV.exists():
            upload_file_to_gcs(FRAME_CSV, f"features/{FRAME_CSV.name}")
            print(f"  Uploaded {FRAME_CSV.name} to GCS")
        if VIDEO_CSV.exists():
            upload_file_to_gcs(VIDEO_CSV, f"features/{VIDEO_CSV.name}")
    except Exception as e:
        print(f"  [warning] GCS upload failed: {e}")


def _aggregate_video_level():
    """Aggregate frame-level predictions to video-level majority vote."""
    if not FRAME_CSV.exists():
        return

    print("\nAggregating video-level predictions...")
    df = pd.read_csv(FRAME_CSV)
    df["video_id"] = df["video_id"].astype(str)

    rows = []
    for vid, grp in df.groupby("video_id"):
        # Majority vote for perspective
        pov_mode = grp["pred_perspective"].mode()
        pov_pred = pov_mode.iloc[0] if len(pov_mode) > 0 else "NA"
        pov_conf = grp["perspective_conf"].astype(float).mean()

        # Majority vote for distance
        dist_mode = grp["pred_distance"].mode()
        dist_pred = dist_mode.iloc[0] if len(dist_mode) > 0 else "NA"
        dist_conf = grp["distance_conf"].astype(float).mean()

        rows.append([vid, pov_pred, f"{pov_conf:.4f}", dist_pred, f"{dist_conf:.4f}", len(grp)])

    video_df = pd.DataFrame(rows, columns=VIDEO_HEADER)
    video_df.to_csv(VIDEO_CSV, index=False)
    print(f"  Video-level predictions: {len(video_df)} videos → {VIDEO_CSV}")

    _upload_checkpoint()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch inference on TikTok videos")
    parser.add_argument("--pov-weights", required=True, help="Path to POV model weights (.pth)")
    parser.add_argument("--dist-weights", required=True, help="Path to distance model weights (.pth)")
    parser.add_argument("--workers", type=int, default=4, help="Number of download workers")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU batch size")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Videos between GCS uploads")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos (0=all)")
    args = parser.parse_args()

    # List all videos in GCS
    print("Listing videos in GCS...")
    from shared.gcs_utils import list_video_blobs
    from shared.config import GCS_VIDEO_PREFIX
    blobs = list_video_blobs(prefix=GCS_VIDEO_PREFIX)
    print(f"  Found {len(blobs)} videos in GCS")

    if args.limit > 0:
        blobs = blobs[:args.limit]
        print(f"  Limited to {len(blobs)} videos")

    run_inference(
        pov_weights=args.pov_weights,
        dist_weights=args.dist_weights,
        blob_names=blobs,
        workers=args.workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
