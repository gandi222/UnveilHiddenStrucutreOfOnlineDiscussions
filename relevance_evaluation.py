"""
relevance_evaluation.py — evaluate continuous relevance scores for strategy D.

Compares the model's predicted relevance score (pred_relevance) against
human-labeled relevance scores (relevance human labeled) using three metrics:
MAE, Pearson correlation, and L2 distance.

Run standalone with:  python relevance_evaluation.py [path/to/results2.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluation import print_section

RESULTS_CSV = "results2.csv"


def compute_relevance_metrics(df: pd.DataFrame) -> dict:
    """Compute MAE, Pearson correlation, and L2 distance between pred_relevance
    and relevance human labeled. Returns a dict with all results and the sample size."""
    sub = df[["pred_relevance", "relevance human labeled"]].dropna()
    pred = sub["pred_relevance"].to_numpy(dtype=float)
    human = sub["relevance human labeled"].to_numpy(dtype=float)
    n = len(pred)

    # MAE — Mean Absolute Error: average of |pred - human| across all pairs.
    # Gives a point-wise measure of how far off the model is on average.
    # A score of 0.0 means perfect agreement; higher values mean larger typical error.
    mae = float(np.mean(np.abs(pred - human)))

    # Pearson r — linear correlation between pred_relevance and human scores.
    # Ranges from -1 (perfect inverse) to +1 (perfect agreement).
    # Captures whether the model's scoring *trends* match human judgments even
    # when the absolute values differ (e.g., model is consistently 0.2 too high).
    pearson_r, pearson_p = stats.pearsonr(pred, human)

    # L2 distance — Euclidean distance between the two score vectors.
    # Formula: sqrt( sum( (pred_i - human_i)^2 ) )
    # Larger errors are penalized quadratically before summing, so a single
    # big outlier has far more weight than several small deviations of equal
    # total magnitude. Complements MAE by exposing whether errors are spread
    # evenly or dominated by a few extreme cases.
    l2_distance = float(np.sqrt(np.sum((pred - human) ** 2)))

    return {
        "n": n,
        "mae": mae,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "l2_distance": l2_distance,
        "pred_mean": float(pred.mean()),
        "human_mean": float(human.mean()),
        "pred_std": float(pred.std()),
        "human_std": float(human.std()),
    }


def print_relevance_results(csv_path) -> None:
    """Load the results CSV, filter to strategy D, and print relevance metrics."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    d_df = df[df["strategy"] == "D"].copy()
    if d_df.empty:
        print("No strategy D rows found — skipping relevance evaluation.")
        return

    if "relevance human labeled" not in d_df.columns:
        print("Column 'relevance human labeled' not found — skipping relevance evaluation.")
        return

    print_section("Strategy D — Relevance Score Evaluation")

    m = compute_relevance_metrics(d_df)

    print(f"\n  Pairs evaluated: {m['n']}")
    print(f"\n  Score distributions:")
    print(f"    {'':25s} {'Mean':>8} {'Std':>8}")
    print(f"    {'-'*42}")
    print(f"    {'pred_relevance':25s} {m['pred_mean']:>8.4f} {m['pred_std']:>8.4f}")
    print(f"    {'relevance human labeled':25s} {m['human_mean']:>8.4f} {m['human_std']:>8.4f}")

    print(f"\n  Metrics:")
    print(f"    MAE          = {m['mae']:.4f}")
    print(f"      (mean |pred - human| over {m['n']} pairs)")
    print(f"    Pearson r    = {m['pearson_r']:.4f}  (p = {m['pearson_p']:.4f}"
          f"{'  *significant*' if m['pearson_p'] < 0.05 else '  not significant'})")
    print(f"      (linear correlation between model and human scores)")
    print(f"    L2 distance  = {m['l2_distance']:.4f}")
    print(f"      (sqrt(sum((pred - human)^2))  — larger errors penalized more)")

    print()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(RESULTS_CSV)
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    print_relevance_results(path)
