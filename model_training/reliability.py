"""
Inter-coder reliability (Cohen's kappa) for doubly-coded frames.

Finds frames annotated by both coders and computes agreement statistics
for the perspective and distance tasks.
"""

from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_cohens_kappa(annotations_df: pd.DataFrame, task: str) -> Dict:
    """
    Compute Cohen's kappa for *task* on doubly-coded frames.

    A frame is "doubly coded" when two different annotators have labeled the
    same (video_id, frame_index) pair.

    Args:
        annotations_df: Full annotations DataFrame.
        task: Column name ('perspective' or 'distance').

    Returns:
        Dict with keys: kappa, n_pairs, agreement, coders.
        If fewer than 2 coders or no overlap, returns kappa=None.
    """
    df = annotations_df.copy()
    df["video_id"] = df["video_id"].astype(str)

    # Drop rows with missing task labels
    df = df.dropna(subset=[task])
    if task == "distance":
        # Exclude NA values for agreement calculation
        df = df[df[task] != "NA"]
    if task == "perspective":
        df = df[df[task] != "NA"]

    coders = sorted(df["annotator"].unique())
    if len(coders) < 2:
        return {"kappa": None, "n_pairs": 0, "agreement": None, "coders": coders}

    # Pivot to get one column per coder
    # Keep only pairs where both coders have a label
    df["frame_key"] = df["video_id"] + "_" + df["frame_index"].astype(int).astype(str)

    pairs = []
    for key, group in df.groupby("frame_key"):
        coder_labels = {}
        for _, row in group.iterrows():
            coder_labels[row["annotator"]] = row[task]
        if len(coder_labels) >= 2:
            # Take the first two coders alphabetically for consistency
            c1, c2 = coders[0], coders[1]
            if c1 in coder_labels and c2 in coder_labels:
                pairs.append((coder_labels[c1], coder_labels[c2]))

    if len(pairs) < 2:
        return {"kappa": None, "n_pairs": len(pairs), "agreement": None, "coders": coders}

    labels_1 = [p[0] for p in pairs]
    labels_2 = [p[1] for p in pairs]

    kappa = cohen_kappa_score(labels_1, labels_2)
    agreement = sum(1 for a, b in pairs if a == b) / len(pairs)

    return {
        "kappa": round(kappa, 4),
        "n_pairs": len(pairs),
        "agreement": round(agreement, 4),
        "coders": coders,
    }
