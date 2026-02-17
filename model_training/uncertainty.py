"""
Uncertainty scoring for the active-learning frame selection.

Scores each unlabeled (video_id, frame_index) pair using the entropy of
both the POV and distance model predictions.  High combined entropy means
the models are uncertain on at least one task — these frames are the most
informative to annotate next.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model_training.dataset import load_frame


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy: H(p) = -sum(p_i * log(p_i)), with 0*log(0)=0."""
    probs = probs.clip(1e-12, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def score_unlabeled_pool(
    pov_model: nn.Module,
    dist_model: nn.Module,
    unlabeled_pairs: List[Tuple[str, int]],
    video_lookup: Dict[str, str],
    transform: transforms.Compose,
    device: torch.device,
    al_round: int = 0,
) -> List[Tuple[str, int, int, float]]:
    """
    Score each unlabeled frame by combined prediction entropy.

    Args:
        pov_model: Trained POV model (eval mode).
        dist_model: Trained distance model (eval mode).
        unlabeled_pairs: List of (video_id, frame_index) to score.
        video_lookup: {video_id: gcs_path}.
        transform: Eval transform (Resize + Normalize).
        device: Torch device.
        al_round: Active-learning round number (stored in output).

    Returns:
        List of (video_id, frame_index, round, combined_entropy)
        sorted by combined_entropy descending (most uncertain first).
    """
    pov_model.eval()
    dist_model.eval()

    results: List[Tuple[str, int, int, float]] = []

    for video_id, frame_index in unlabeled_pairs:
        img = load_frame(video_id, frame_index, video_lookup)
        if img is None:
            continue

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pov_out = pov_model(tensor)
            dist_out = dist_model(tensor)

        pov_probs = torch.softmax(pov_out, dim=1).cpu().numpy()[0]
        dist_probs = torch.softmax(dist_out, dim=1).cpu().numpy()[0]

        score = _entropy(pov_probs) + _entropy(dist_probs)
        results.append((video_id, frame_index, al_round, score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results
