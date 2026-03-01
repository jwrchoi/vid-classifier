"""
Fine-tune ResNet-50 v7 → v8 using agreed ICR frames with LOOCV.

Downloads annotations.csv, extracts doubly-coded frames where both coders
agree, then runs Leave-One-Out Cross-Validation to compare v7 baseline
accuracy against a fine-tuned v8 candidate.  Only saves v8 weights when
they improve over v7.

Usage:
    python -m model_training.fine_tune_icr
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training.config import (
    ANNOTATIONS_FILE,
    GCS_BUCKET_NAME,
    MODEL_CONFIGS,
    V8_OUTPUT_DIR,
    find_models_dir,
)
from model_training.dataset import (
    build_video_lookup,
    get_eval_transform,
    get_train_transform,
    load_frame,
)
from model_training.trainer import build_model, get_device

# ---------------------------------------------------------------------------
# LOOCV hyper-parameters (small-data regime)
# ---------------------------------------------------------------------------
EPOCHS = 15
LR = 1e-5
PATIENCE = 3  # epochs without training-loss improvement


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_annotations() -> pd.DataFrame:
    """Download annotations.csv from GCS (the authoritative copy).

    Falls back to the local file only if GCS is unreachable.
    """
    try:
        from shared.gcs_utils import get_gcs_client
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("annotations/annotations.csv")
        import io
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        print(f"  Downloaded annotations from GCS ({len(df)} rows)")
    except Exception as e:
        print(f"  GCS download failed ({e}), falling back to local file")
        df = pd.read_csv(ANNOTATIONS_FILE)
    df["video_id"] = df["video_id"].astype(str)
    return df


def extract_agreed_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where two coders annotated the same frame and agreed."""
    # Group by (video_id, frame_index) — need at least 2 coders
    key_cols = ["video_id", "frame_index"]
    groups = df.groupby(key_cols)

    agreed_rows = []
    for (vid, fi), grp in groups:
        if len(grp) < 2:
            continue
        # Check agreement on perspective and distance separately
        perspectives = grp["perspective"].dropna().unique()
        distances = grp["distance"].dropna().unique()
        row = {"video_id": vid, "frame_index": fi}
        if len(perspectives) == 1 and perspectives[0] != "NA":
            row["perspective"] = perspectives[0]
        if len(distances) == 1 and distances[0] != "NA":
            row["distance"] = distances[0]
        if "perspective" in row or "distance" in row:
            agreed_rows.append(row)

    return pd.DataFrame(agreed_rows)


# ---------------------------------------------------------------------------
# Frame pre-loading (avoids re-downloading inside the LOOCV loop)
# ---------------------------------------------------------------------------

def preload_frames(
    gold: pd.DataFrame,
    video_lookup: Dict[str, str],
) -> Dict[Tuple[str, int], "Image.Image"]:
    """Load all gold-standard frames into memory."""
    from PIL import Image
    cache: Dict[Tuple[str, int], Image.Image] = {}
    total = len(gold)
    for i, row in gold.iterrows():
        vid = str(row["video_id"])
        fi = int(row["frame_index"])
        img = load_frame(vid, fi, video_lookup)
        if img is None:
            from model_training.config import IMG_SIZE
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        cache[(vid, fi)] = img
        done = len(cache)
        if done % 10 == 0 or done == total:
            print(f"  Cached {done}/{total} frames")
    return cache


def frames_to_tensors(
    gold: pd.DataFrame,
    task: str,
    classes: List[str],
    frame_cache: Dict,
    transform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert gold frames into (N, C, H, W) tensor + label tensor."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    imgs, labels = [], []
    for _, row in gold.iterrows():
        vid = str(row["video_id"])
        fi = int(row["frame_index"])
        img = frame_cache[(vid, fi)]
        imgs.append(transform(img))
        labels.append(class_to_idx[row[task]])
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def freeze_early_layers(model: nn.Module) -> None:
    """Freeze everything except layer4 and fc."""
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return weights


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------

def run_loocv(
    task: str,
    gold: pd.DataFrame,
    frame_cache: Dict,
    device: torch.device,
) -> Dict:
    """Run LOOCV for a single task, comparing v7 baseline vs v8 fine-tuned."""
    cfg = MODEL_CONFIGS[task]
    classes = cfg["classes"]
    num_classes = cfg["num_classes"]
    v7_path = str(find_models_dir() / cfg["v7_filename"])

    print(f"\n{'='*60}")
    print(f"LOOCV for {task.upper()} — {len(gold)} agreed frames")
    print(f"Classes: {classes}")
    print(f"V7 weights: {v7_path}")
    print(f"{'='*60}")

    # Pre-compute all tensors with eval transform (for v7 baseline predictions)
    eval_tf = get_eval_transform()
    all_imgs_eval, all_labels = frames_to_tensors(gold, task, classes, frame_cache, eval_tf)

    # Pre-compute all tensors with train transform (for v8 training folds)
    train_tf = get_train_transform()
    all_imgs_train, _ = frames_to_tensors(gold, task, classes, frame_cache, train_tf)

    N = len(gold)
    v7_preds = []
    v8_preds = []
    true_labels = all_labels.numpy().tolist()

    for i in range(N):
        # --- V7 baseline: just predict the held-out frame ---
        v7_model = build_model(num_classes, v7_path, device)
        v7_model.eval()
        with torch.no_grad():
            held_out_img = all_imgs_eval[i].unsqueeze(0).to(device)
            v7_out = v7_model(held_out_img)
            v7_pred = v7_out.argmax(dim=1).item()
        v7_preds.append(v7_pred)

        # --- V8 candidate: fine-tune on N-1, predict held-out ---
        train_mask = torch.ones(N, dtype=torch.bool)
        train_mask[i] = False

        train_imgs = all_imgs_train[train_mask]
        train_labels = all_labels[train_mask]

        class_weights = compute_class_weights(train_labels, num_classes).to(device)

        v8_model = build_model(num_classes, v7_path, device)
        freeze_early_layers(v8_model)
        v8_model.train()

        trainable_params = [p for p in v8_model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=LR)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_ds = TensorDataset(train_imgs, train_labels)
        loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)

        best_loss = float("inf")
        no_improve = 0
        for epoch in range(EPOCHS):
            v8_model.train()
            for batch_imgs, batch_labels in loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                out = v8_model(batch_imgs)
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer.step()

            # Simple patience on training loss
            with torch.no_grad():
                v8_model.eval()
                all_train_out = v8_model(train_imgs.to(device))
                epoch_loss = criterion(all_train_out, train_labels.to(device)).item()
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

        # Predict held-out
        v8_model.eval()
        with torch.no_grad():
            held_eval = all_imgs_eval[i].unsqueeze(0).to(device)
            v8_out = v8_model(held_eval)
            v8_pred = v8_out.argmax(dim=1).item()
        v8_preds.append(v8_pred)

        label_name = classes[true_labels[i]]
        v7_name = classes[v7_pred]
        v8_name = classes[v8_pred]
        status = ""
        if v7_pred != true_labels[i] and v8_pred == true_labels[i]:
            status = " [v8 FIX]"
        elif v7_pred == true_labels[i] and v8_pred != true_labels[i]:
            status = " [v8 REGRESS]"
        print(f"  Fold {i+1:3d}/{N}  true={label_name:<12s}  v7={v7_name:<12s}  v8={v8_name:<12s}{status}")

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    v7_acc = accuracy_score(true_labels, v7_preds)
    v8_acc = accuracy_score(true_labels, v8_preds)
    v7_f1 = f1_score(true_labels, v7_preds, average="macro", zero_division=0)
    v8_f1 = f1_score(true_labels, v8_preds, average="macro", zero_division=0)

    result = {
        "task": task,
        "n_samples": N,
        "classes": classes,
        "v7_accuracy": round(v7_acc, 4),
        "v8_accuracy": round(v8_acc, 4),
        "v7_macro_f1": round(v7_f1, 4),
        "v8_macro_f1": round(v8_f1, 4),
        "v7_preds": v7_preds,
        "v8_preds": v8_preds,
        "true_labels": true_labels,
        "v8_improves": v8_f1 > v7_f1,
    }

    print(f"\n  {'Metric':<20s} {'v7':>10s} {'v8':>10s} {'delta':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<20s} {v7_acc:>10.4f} {v8_acc:>10.4f} {v8_acc - v7_acc:>+10.4f}")
    print(f"  {'Macro F1':<20s} {v7_f1:>10.4f} {v8_f1:>10.4f} {v8_f1 - v7_f1:>+10.4f}")

    return result


# ---------------------------------------------------------------------------
# Full v8 training (if LOOCV shows improvement)
# ---------------------------------------------------------------------------

def train_full_v8(
    task: str,
    gold: pd.DataFrame,
    frame_cache: Dict,
    device: torch.device,
) -> Path:
    """Train v8 on ALL agreed frames (no held-out) and save weights."""
    cfg = MODEL_CONFIGS[task]
    classes = cfg["classes"]
    num_classes = cfg["num_classes"]
    v7_path = str(find_models_dir() / cfg["v7_filename"])

    train_tf = get_train_transform()
    all_imgs, all_labels = frames_to_tensors(gold, task, classes, frame_cache, train_tf)

    class_weights = compute_class_weights(all_labels, num_classes).to(device)

    model = build_model(num_classes, v7_path, device)
    freeze_early_layers(model)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ds = TensorDataset(all_imgs, all_labels)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=True)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            out = model(batch_imgs)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(all_imgs.to(device))
            epoch_loss = criterion(out, all_labels.to(device)).item()

        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}")
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Converged at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    V8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_prefix = "pov" if task == "perspective" else "distance"
    out_path = V8_OUTPUT_DIR / f"{task_prefix}_resnet50_v8.pth"
    torch.save(model.state_dict(), out_path)
    print(f"  Saved v8 weights → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Device: {device}")

    # 1. Load annotations
    print("\nLoading annotations...")
    df = download_annotations()
    print(f"  Total rows: {len(df)}")

    # 2. Extract agreed ICR frames
    print("\nExtracting agreed ICR frames...")
    gold = extract_agreed_frames(df)
    print(f"  Agreed frames: {len(gold)}")

    if "perspective" in gold.columns:
        pov_gold = gold.dropna(subset=["perspective"])
        print(f"  With agreed POV: {len(pov_gold)}")
        if len(pov_gold) > 0:
            print(f"  POV distribution: {pov_gold['perspective'].value_counts().to_dict()}")
    else:
        pov_gold = pd.DataFrame()

    if "distance" in gold.columns:
        dist_gold = gold.dropna(subset=["distance"])
        print(f"  With agreed distance: {len(dist_gold)}")
        if len(dist_gold) > 0:
            print(f"  Distance distribution: {dist_gold['distance'].value_counts().to_dict()}")
    else:
        dist_gold = pd.DataFrame()

    if len(pov_gold) == 0 and len(dist_gold) == 0:
        print("\nNo agreed frames found. Exiting.")
        return

    # 3. Build video lookup and preload frames
    print("\nBuilding video lookup...")
    video_lookup = build_video_lookup()

    # Collect all unique (video_id, frame_index) pairs needed
    all_frames = pd.concat([pov_gold, dist_gold]).drop_duplicates(
        subset=["video_id", "frame_index"]
    )
    print(f"\nPreloading {len(all_frames)} unique frames...")
    frame_cache = preload_frames(all_frames, video_lookup)

    # 4. Run LOOCV for each task
    results = {}

    if len(pov_gold) >= 3:
        results["perspective"] = run_loocv("perspective", pov_gold, frame_cache, device)
    else:
        print(f"\nSkipping POV LOOCV — only {len(pov_gold)} agreed frames (need ≥3)")

    if len(dist_gold) >= 3:
        results["distance"] = run_loocv("distance", dist_gold, frame_cache, device)
    else:
        print(f"\nSkipping distance LOOCV — only {len(dist_gold)} agreed frames (need ≥3)")

    # 5. Save comparison report
    V8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = V8_OUTPUT_DIR / "v8_comparison.json"

    # Strip numpy types for JSON serialization
    report = {}
    for task, r in results.items():
        report[task] = {k: v for k, v in r.items()}

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComparison report → {report_path}")

    # 6. Train full v8 if improvement found
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task, r in results.items():
        improved = r["v8_improves"]
        tag = "IMPROVED" if improved else "NO IMPROVEMENT"
        print(f"  {task:<15s}  v7_f1={r['v7_macro_f1']:.4f}  v8_f1={r['v8_macro_f1']:.4f}  [{tag}]")

        if improved:
            task_gold = pov_gold if task == "perspective" else dist_gold
            print(f"\n  Training full v8 {task} model on all {len(task_gold)} agreed frames...")
            train_full_v8(task, task_gold, frame_cache, device)
        else:
            print(f"  → Keeping v7 for {task}")

    print("\nDone.")


if __name__ == "__main__":
    main()
