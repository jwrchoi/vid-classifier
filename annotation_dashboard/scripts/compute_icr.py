"""
Compute inter-coder reliability (ICR) for perspective and distance annotations.

Metrics reported:
  - Percent agreement
  - Cohen's kappa       (pairwise; excludes frames where either coder has no annotation)
  - Krippendorff's alpha (nominal; same exclusion for 2-rater case)

When no_human_visible=True, perspective and distance are treated as "NA" (valid label).

Output:
  data/icr_report.csv  — one row per overlapping frame, both coders side by side
  stdout               — agreement statistics + disagreement listings

Usage:
    python annotation_dashboard/scripts/compute_icr.py [--coder1 Soojin] [--coder2 Janice]
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage

BUCKET = "vid-classifier-db"
ANNOTATIONS_PATH = "annotations/annotations.csv"
REPORT_PATH = Path(__file__).parents[2] / "data" / "icr_report.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_annotations() -> pd.DataFrame:
    client = storage.Client()
    blob = client.bucket(BUCKET).blob(ANNOTATIONS_PATH)
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    df["video_id"] = df["video_id"].astype(str)
    # Normalise no_human_visible to bool regardless of how it was stored
    df["no_human_visible"] = df["no_human_visible"].map(
        lambda v: str(v).strip().lower() == "true"
    )
    return df


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

def effective_label(row: pd.Series, field: str) -> str | None:
    """Return 'NA' when no_human_visible=True; otherwise the coder's label."""
    if row["no_human_visible"]:
        return "NA"
    return row[field] if pd.notna(row[field]) else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def percent_agreement(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    return (a[mask] == b[mask]).mean()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return float("nan")
    categories = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in categories)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(a: pd.Series, b: pd.Series) -> float:
    """
    Krippendorff's alpha for nominal data with exactly 2 raters.

    For 2 raters, units where either annotation is missing cannot contribute
    to the coincidence matrix, so they are excluded.  This is equivalent to
    the standard Krippendorff treatment for missing data.
    """
    mask = a.notna() & b.notna()
    a_obs = a[mask].values
    b_obs = b[mask].values
    n_units = len(a_obs)
    if n_units < 2:
        return float("nan")

    categories = sorted(set(a_obs) | set(b_obs))
    cat_idx = {c: i for i, c in enumerate(categories)}
    nc = len(categories)

    # Coincidence matrix (symmetric):
    #   diagonal entry o[v,v] = 2 * count(both say v)
    #   off-diagonal   o[v,k] = count(a=v,b=k) + count(a=k,b=v)
    o = np.zeros((nc, nc))
    for av, bv in zip(a_obs, b_obs):
        i, j = cat_idx[av], cat_idx[bv]
        if i == j:
            o[i, i] += 2
        else:
            o[i, j] += 1
            o[j, i] += 1

    n = o.sum()           # = 2 * n_units
    n_v = o.sum(axis=1)   # marginal counts per category

    # Observed disagreement = 1 − (diagonal proportion)
    D_o = 1 - np.trace(o) / n

    # Expected disagreement
    D_e = (n ** 2 - (n_v ** 2).sum()) / (n * (n - 1))

    if D_e == 0:
        return 1.0
    return 1 - D_o / D_e


# ---------------------------------------------------------------------------
# Wide-format alignment
# ---------------------------------------------------------------------------

def build_wide_table(df: pd.DataFrame, coder1: str, coder2: str) -> pd.DataFrame:
    """Return one row per (video_id, frame_index) coded by both coders."""
    df = df.copy()
    df["eff_perspective"] = df.apply(lambda r: effective_label(r, "perspective"), axis=1)
    df["eff_distance"] = df.apply(lambda r: effective_label(r, "distance"), axis=1)

    def coder_slice(name: str) -> pd.DataFrame:
        sub = df[df["annotator"].str.lower() == name.lower()].copy()
        sub = sub.dropna(subset=["frame_index"])
        sub["frame_index"] = sub["frame_index"].astype(int)
        return sub[["video_id", "frame_index", "no_human_visible",
                     "eff_perspective", "eff_distance"]].set_index(["video_id", "frame_index"])

    df1 = coder_slice(coder1)
    df2 = coder_slice(coder2)

    joined = df1.join(df2, lsuffix=f"_{coder1}", rsuffix=f"_{coder2}", how="inner")
    joined["perspective_agree"] = (
        joined[f"eff_perspective_{coder1}"] == joined[f"eff_perspective_{coder2}"]
    )
    joined["distance_agree"] = (
        joined[f"eff_distance_{coder1}"] == joined[f"eff_distance_{coder2}"]
    )
    return joined.reset_index()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_stats(wide: pd.DataFrame, coder1: str, coder2: str) -> None:
    for field, agree_col in [("perspective", "perspective_agree"),
                              ("distance", "distance_agree")]:
        a = wide[f"eff_{field}_{coder1}"]
        b = wide[f"eff_{field}_{coder2}"]
        n_both = (a.notna() & b.notna()).sum()
        pct = percent_agreement(a, b)
        kappa = cohen_kappa(a, b)
        alpha = krippendorff_alpha_nominal(a, b)

        print(f"\n{'='*54}")
        print(f"  {field.upper()}")
        print(f"{'='*54}")
        print(f"  Frames with both coders coded : {n_both}")
        print(f"  Percent agreement             : {pct:.1%}")
        print(f"  Cohen's kappa                 : {kappa:.3f}")
        print(f"  Krippendorff's alpha (nominal): {alpha:.3f}")


def print_disagreements(wide: pd.DataFrame, coder1: str, coder2: str) -> None:
    for field, agree_col in [("perspective", "perspective_agree"),
                              ("distance", "distance_agree")]:
        disagree = wide[~wide[agree_col]][
            ["video_id", "frame_index",
             f"eff_{field}_{coder1}", f"eff_{field}_{coder2}"]
        ].rename(columns={
            f"eff_{field}_{coder1}": coder1,
            f"eff_{field}_{coder2}": coder2,
        })
        print(f"\n{'='*54}")
        print(f"  {field.upper()} DISAGREEMENTS  ({len(disagree)} frames)")
        print(f"{'='*54}")
        if disagree.empty:
            print("  (none)")
        else:
            print(disagree.to_string(index=False))


def save_csv(wide: pd.DataFrame, coder1: str, coder2: str) -> None:
    out = wide.rename(columns={
        f"eff_perspective_{coder1}": f"{coder1}_perspective",
        f"eff_perspective_{coder2}": f"{coder2}_perspective",
        f"eff_distance_{coder1}":    f"{coder1}_distance",
        f"eff_distance_{coder2}":    f"{coder2}_distance",
        f"no_human_visible_{coder1}": f"{coder1}_no_human",
        f"no_human_visible_{coder2}": f"{coder2}_no_human",
    })[[
        "video_id", "frame_index",
        f"{coder1}_no_human",      f"{coder2}_no_human",
        f"{coder1}_perspective",   f"{coder2}_perspective", "perspective_agree",
        f"{coder1}_distance",      f"{coder2}_distance",    "distance_agree",
    ]]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coder1", default="Soojin")
    parser.add_argument("--coder2", default="Janice")
    args = parser.parse_args()

    print("Fetching annotations from GCS …")
    df = fetch_annotations()

    wide = build_wide_table(df, args.coder1, args.coder2)
    print(f"Overlapping frames (coded by both): {len(wide)}")

    print_stats(wide, args.coder1, args.coder2)
    print_disagreements(wide, args.coder1, args.coder2)
    save_csv(wide, args.coder1, args.coder2)


if __name__ == "__main__":
    main()
