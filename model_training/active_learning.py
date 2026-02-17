"""
Active-learning orchestrator.
=============================

Usage:
    python model_training/active_learning.py --round 0 --top-k 50
    python model_training/active_learning.py --round 1 --epochs 30

Flow for each round:
1.  Create  rounds/round_NN/  directory.
2.  Load annotations.csv.
3.  Round 0 special: create a fixed test set (stratified 20 %) and save to
    rounds/test_set.csv.  Subsequent rounds reuse this file.
4.  Split remaining labeled data into train / val (80 / 20).
5.  Compute inter-coder reliability (Cohen's kappa).
6.  Resolve starting weights (round 0 → v7 models; round N → round N-1 best).
7.  Train POV model → save round_NN/pov_resnet50.pth.
8.  Train distance model → save round_NN/distance_resnet50.pth.
9.  Score unlabeled pool by combined entropy.
10. Select top-K → write round_NN/queue.csv + copy to data/queue.csv.
11. Save round_NN/metrics.json.
12. Print summary + stopping guidance.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training.config import (
    ANNOTATIONS_FILE,
    BATCH_SIZE,
    DASHBOARD_QUEUE_PATH,
    EPOCHS,
    FRAMES_PER_VIDEO,
    LEARNING_RATE,
    MODEL_CONFIGS,
    PATIENCE,
    ROUNDS_DIR,
    TEST_FRACTION,
    TOP_K,
    find_models_dir,
)
from model_training.dataset import (
    FrameDataset,
    build_video_lookup,
    get_eval_transform,
    get_train_transform,
)
from model_training.reliability import compute_cohens_kappa
from model_training.trainer import evaluate, get_device, train_model
from model_training.uncertainty import score_unlabeled_pool


# =============================================================================
# HELPERS
# =============================================================================

def _round_dir(round_num: int) -> Path:
    return ROUNDS_DIR / f"round_{round_num:02d}"


def _stratified_split(df: pd.DataFrame, task: str, fraction: float, seed: int = 42):
    """Split *df* into (subset, remainder) with stratification on *task*."""
    from sklearn.model_selection import train_test_split

    # Only stratify on rows that have a valid label for this task
    valid = df[df[task].notna() & (df[task] != "NA")]
    if len(valid) < 5:
        # Too few samples for stratification — random split
        subset = valid.sample(frac=fraction, random_state=seed)
        remainder = valid.drop(subset.index)
        return subset, remainder

    try:
        subset, remainder = train_test_split(
            valid, test_size=1 - fraction, stratify=valid[task], random_state=seed
        )
    except ValueError:
        # Fallback if a class has < 2 members
        subset = valid.sample(frac=fraction, random_state=seed)
        remainder = valid.drop(subset.index)
    return subset, remainder


def _build_test_set(annotations: pd.DataFrame, test_frac: float) -> pd.DataFrame:
    """Create a fixed test set from seed annotations (run once in round 0)."""
    test_path = ROUNDS_DIR / "test_set.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        df["video_id"] = df["video_id"].astype(str)
        return df

    # Use perspective for stratification (most balanced)
    test_df, _ = _stratified_split(annotations, "perspective", test_frac, seed=42)
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)
    print(f"  Created fixed test set: {len(test_df)} samples → {test_path}")
    return test_df


def _resolve_weights(round_num: int, task_key: str):
    """
    Return the path to starting weights for this round.

    Round 0: v7 models from MODELS_DIR.
    Round N: best model from round N-1.
    """
    cfg = MODEL_CONFIGS[task_key]
    if round_num == 0:
        models_dir = find_models_dir()
        path = models_dir / cfg["v7_filename"]
        if path.exists():
            return str(path)
        print(f"  WARNING: v7 weights not found at {path}; training from scratch")
        return None
    else:
        prev_dir = _round_dir(round_num - 1)
        path = prev_dir / cfg["output_filename"]
        if path.exists():
            return str(path)
        print(f"  WARNING: previous round weights not found at {path}; training from scratch")
        return None


def _build_unlabeled_pool(
    annotations: pd.DataFrame,
    video_lookup: dict,
) -> list:
    """
    Return list of (video_id, frame_index) pairs NOT in annotations.
    """
    annotated = set()
    for _, row in annotations.iterrows():
        vid = str(row["video_id"])
        fi = row.get("frame_index")
        if pd.notna(fi):
            annotated.add((vid, int(fi)))

    pool = []
    for video_id in video_lookup:
        for fi in range(FRAMES_PER_VIDEO):
            if (video_id, fi) not in annotated:
                pool.append((video_id, fi))
    return pool


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Active-learning round orchestrator")
    parser.add_argument("--round", type=int, required=True, help="Round number (0-based)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Frames to select")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()

    round_num = args.round
    rdir = _round_dir(round_num)
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"=== Active Learning Round {round_num} ===")
    print(f"  Output: {rdir}")

    device = get_device()
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load annotations
    # ------------------------------------------------------------------
    if not ANNOTATIONS_FILE.exists():
        print(f"ERROR: annotations file not found: {ANNOTATIONS_FILE}")
        sys.exit(1)

    all_ann = pd.read_csv(ANNOTATIONS_FILE)
    all_ann["video_id"] = all_ann["video_id"].astype(str)
    print(f"  Annotations loaded: {len(all_ann)} rows")

    video_lookup = build_video_lookup()

    # ------------------------------------------------------------------
    # 2. Test set (fixed across all rounds)
    # ------------------------------------------------------------------
    test_df = _build_test_set(all_ann, args.test_fraction)
    test_keys = set(
        zip(test_df["video_id"].astype(str), test_df["frame_index"].astype(int))
    )

    # Remaining = train pool (exclude test rows)
    train_pool = all_ann[
        ~all_ann.apply(
            lambda r: (str(r["video_id"]), int(r["frame_index"])) in test_keys
            if pd.notna(r.get("frame_index")) else False,
            axis=1,
        )
    ].copy()

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    if len(train_pool) > 10:
        val_df, train_df = _stratified_split(train_pool, "perspective", 0.2, seed=round_num)
    else:
        train_df = train_pool
        val_df = train_pool  # tiny dataset fallback

    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ------------------------------------------------------------------
    # 4. Inter-coder reliability
    # ------------------------------------------------------------------
    kappa_results = {}
    for task in ["perspective", "distance"]:
        kappa_results[task] = compute_cohens_kappa(all_ann, task)
        k = kappa_results[task]
        if k["kappa"] is not None:
            print(f"  Kappa ({task}): {k['kappa']:.3f}  (n={k['n_pairs']}, agree={k['agreement']:.2%})")
        else:
            print(f"  Kappa ({task}): not enough overlap (n={k['n_pairs']})")

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    trained_models = {}
    all_metrics = {}
    eval_transform = get_eval_transform()
    train_transform = get_train_transform()

    for task_key, cfg in MODEL_CONFIGS.items():
        print(f"\n--- Training {task_key} ({cfg['num_classes']} classes) ---")

        weights_path = _resolve_weights(round_num, task_key)
        if weights_path:
            print(f"  Starting from: {weights_path}")

        train_ds = FrameDataset(train_df, task_key, cfg["classes"], video_lookup, train_transform)
        val_ds = FrameDataset(val_df, task_key, cfg["classes"], video_lookup, eval_transform)
        test_ds = FrameDataset(test_df, task_key, cfg["classes"], video_lookup, eval_transform)

        if len(train_ds) == 0:
            print(f"  SKIP: no valid training samples for {task_key}")
            continue

        model, metrics = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            num_classes=cfg["num_classes"],
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            patience=PATIENCE,
            pretrained_weights_path=weights_path,
        )

        # Save best model
        save_path = rdir / cfg["output_filename"]
        torch.save(model.state_dict(), save_path)
        print(f"  Saved: {save_path}")
        print(
            f"  Test: acc={metrics['test']['accuracy']:.3f}  "
            f"F1={metrics['test']['macro_f1']:.3f}  "
            f"n={metrics['test']['n_samples']}"
        )

        trained_models[task_key] = model
        all_metrics[task_key] = metrics

    # ------------------------------------------------------------------
    # 6. Score unlabeled pool
    # ------------------------------------------------------------------
    unlabeled = _build_unlabeled_pool(all_ann, video_lookup)
    print(f"\n  Unlabeled pool: {len(unlabeled)} frames")

    queue_rows = []
    if "perspective" in trained_models and "distance" in trained_models and unlabeled:
        print("  Scoring unlabeled pool...")
        scored = score_unlabeled_pool(
            pov_model=trained_models["perspective"],
            dist_model=trained_models["distance"],
            unlabeled_pairs=unlabeled,
            video_lookup=video_lookup,
            transform=eval_transform,
            device=device,
            al_round=round_num,
        )
        # Select top-K
        selected = scored[: args.top_k]
        queue_rows = [
            {"video_id": vid, "frame_index": fi, "round": rnd, "uncertainty_score": sc}
            for vid, fi, rnd, sc in selected
        ]
        print(f"  Selected {len(queue_rows)} frames (top-K={args.top_k})")
    else:
        print("  Skipping uncertainty scoring (missing model or empty pool)")

    # ------------------------------------------------------------------
    # 7. Write queue CSVs
    # ------------------------------------------------------------------
    if queue_rows:
        queue_df = pd.DataFrame(queue_rows)
        # Round-specific copy
        queue_df.to_csv(rdir / "queue.csv", index=False)
        # Dashboard copy
        DASHBOARD_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        queue_df.to_csv(DASHBOARD_QUEUE_PATH, index=False)
        print(f"  Queue written: {rdir / 'queue.csv'} + {DASHBOARD_QUEUE_PATH}")

    # ------------------------------------------------------------------
    # 8. Save metrics
    # ------------------------------------------------------------------
    # Convert numpy values to Python-native for JSON serialization
    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(i) for i in obj]
        return obj

    summary = {
        "round": round_num,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "unlabeled_pool_size": len(unlabeled),
        "selected_frames": len(queue_rows),
        "kappa": _jsonify(kappa_results),
        "models": {},
    }
    for task_key, m in all_metrics.items():
        weights_path = _resolve_weights(round_num, task_key)
        summary["models"][task_key] = {
            "starting_weights": weights_path,
            "epochs_trained": m["epochs_trained"],
            "best_val_loss": m["best_val_loss"],
            "final_train_acc": m["final_train_acc"],
            "final_val_acc": m["final_val_acc"],
            "test_accuracy": m["test"]["accuracy"],
            "test_macro_f1": m["test"]["macro_f1"],
            "test_n_samples": m["test"]["n_samples"],
            "confusion_matrix": m["test"]["confusion_matrix"],
        }

    metrics_path = rdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_jsonify(summary), f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # 9. Stopping guidance
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    for task_key, m in all_metrics.items():
        test_f1 = m["test"]["macro_f1"]
        train_acc = m["final_train_acc"]
        val_acc = m["final_val_acc"]
        print(f"  {task_key}: test_F1={test_f1:.3f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        # Overfitting check
        if train_acc - val_acc > 0.15:
            print(f"    WARNING: overfitting detected (train-val gap = {train_acc - val_acc:.2%})")

    # Compare with previous round
    if round_num > 0:
        prev_metrics_path = _round_dir(round_num - 1) / "metrics.json"
        if prev_metrics_path.exists():
            with open(prev_metrics_path) as f:
                prev = json.load(f)
            print("\n  Comparison with previous round:")
            for task_key in all_metrics:
                if task_key in prev.get("models", {}):
                    prev_f1 = prev["models"][task_key].get("test_macro_f1", 0)
                    curr_f1 = all_metrics[task_key]["test"]["macro_f1"]
                    delta = curr_f1 - prev_f1
                    print(f"    {task_key}: F1 {prev_f1:.3f} → {curr_f1:.3f} (Δ={delta:+.3f})")
                    if abs(delta) < 0.01:
                        print(f"    STOP candidate: improvement < 1% for {task_key}")

    print("\nDone.")


if __name__ == "__main__":
    main()
