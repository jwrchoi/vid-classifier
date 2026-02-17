"""
Fine-tune POV (Perspective) classifier.

Thin wrapper around trainer.train_model() for standalone single-model training.

Usage:
    python model_training/fine_tune_pov.py [--epochs 30] [--lr 1e-4]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_training.config import (
    ANNOTATIONS_FILE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_CONFIGS,
    PATIENCE,
    ROUNDS_DIR,
    find_models_dir,
)
from model_training.dataset import (
    FrameDataset,
    build_video_lookup,
    get_eval_transform,
    get_train_transform,
)
from model_training.trainer import get_device, train_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune POV classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", type=str, default=None, help="Output .pth path")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS["perspective"]
    device = get_device()
    video_lookup = build_video_lookup()

    ann = pd.read_csv(ANNOTATIONS_FILE)
    ann["video_id"] = ann["video_id"].astype(str)

    # Simple 80/10/10 split
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(ann, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_ds = FrameDataset(train_df, "perspective", cfg["classes"], video_lookup, get_train_transform())
    val_ds = FrameDataset(val_df, "perspective", cfg["classes"], video_lookup, get_eval_transform())
    test_ds = FrameDataset(test_df, "perspective", cfg["classes"], video_lookup, get_eval_transform())

    # Start from v7 weights if available
    weights_path = find_models_dir() / cfg["v7_filename"]
    weights = str(weights_path) if weights_path.exists() else None

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
        pretrained_weights_path=weights,
    )

    out = Path(args.output) if args.output else Path("pov_resnet50_finetuned.pth")
    torch.save(model.state_dict(), out)
    print(f"Saved: {out}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.3f}")
    print(f"Test macro F1: {metrics['test']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
