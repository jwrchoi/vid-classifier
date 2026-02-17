"""
Training and evaluation utilities for the active-learning pipeline.

Builds a ResNet-50 matching the dashboard's ModelLoader architecture,
trains with early stopping, and evaluates with sklearn metrics.
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from model_training.config import IMG_SIZE


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:
    """MPS > CUDA > CPU (same priority as the dashboard)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    num_classes: int,
    pretrained_weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create a ResNet-50 with a custom final FC layer.

    Architecture matches ``ModelLoader.load_model`` exactly:
        resnet50(weights=None) -> replace fc -> load state_dict

    Args:
        num_classes: Number of output classes.
        pretrained_weights_path: Path to a .pth file with trained weights.
            If None, starts from random weights (no ImageNet).
        device: Target device.

    Returns:
        Model in eval mode on *device*.
    """
    if device is None:
        device = get_device()

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_ftrs, num_classes)

    if pretrained_weights_path is not None:
        state_dict = torch.load(pretrained_weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model.to(device)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    patience: int = 5,
    pretrained_weights_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Full training loop with early stopping on validation loss.

    Args:
        train_ds, val_ds, test_ds: PyTorch Datasets.
        num_classes: Number of classes.
        epochs: Maximum epochs.
        lr: Learning rate.
        batch_size: Batch size.
        device: Torch device (auto-detected if None).
        patience: Early-stopping patience.
        pretrained_weights_path: Starting weights (.pth).

    Returns:
        (best_model, metrics_dict) where metrics_dict has train/val/test results.
    """
    if device is None:
        device = get_device()

    model = build_model(num_classes, pretrained_weights_path, device)
    model.train()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

        scheduler.step()

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    # Evaluate on test set
    test_metrics = evaluate(model, test_ds, num_classes, device, batch_size)

    metrics = {
        "history": history,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test": test_metrics,
    }

    return model, metrics


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    model: nn.Module,
    dataset: Dataset,
    num_classes: int,
    device: torch.device,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate a model on a dataset.

    Returns accuracy, macro F1, per-class precision/recall/F1,
    and the confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(all_labels),
    }
