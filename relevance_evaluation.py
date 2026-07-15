"""
relevance_evaluation.py — evaluate continuous relevance scores for strategy D.

Compares the model's predicted relevance score (pred_relevance) against one or
two human-labeled relevance score columns using MAE, RMSE, Pearson r, and L2
distance.  When a second labeler column ("relevance human labeled #2") is
present the script also reports inter-rater agreement between the two humans.

Run standalone with:  python relevance_evaluation.py [path/to/results.csv]
Default file:         results/NR_WebDataset_D_Batch10_zeroShot_HumanLabeled_2.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluation import print_section

RESULTS_CSV = "results/NR_WebDataset_D_Batch10_zeroShot_HumanLabeled_2.csv"

COL_PRED   = "pred_relevance"
COL_H1     = "relevance human labeled"
COL_H2     = "relevance human labeled #2"


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute pairwise metrics between two equal-length float arrays."""
    n = len(a)

    # MAE — Mean Absolute Error: average of |a - b| across all pairs.
    mae = float(np.mean(np.abs(a - b)))

    # RMSE — Root Mean Square Error: penalizes larger errors more than MAE.
    # If RMSE >> MAE a few large outliers dominate; if RMSE ≈ MAE errors are even.
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))

    # Pearson r — linear correlation; captures trend agreement independent of
    # absolute offset (e.g. one rater consistently 0.2 higher still gives r≈1).
    pearson_r, pearson_p = stats.pearsonr(a, b)

    # L2 distance — Euclidean distance; single large outlier has outsized weight
    # compared to many small errors of equal total magnitude.
    l2 = float(np.sqrt(np.sum((a - b) ** 2)))

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "l2_distance": l2,
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "a_std": float(a.std()),
        "b_std": float(b.std()),
    }


def compute_relevance_metrics(df: pd.DataFrame) -> dict:
    """Return metrics for labeler 1 (and optionally labeler 2 + inter-rater)."""
    result = {}

    sub1 = df[[COL_PRED, COL_H1]].dropna()
    pred1 = sub1[COL_PRED].to_numpy(dtype=float)
    h1    = sub1[COL_H1].to_numpy(dtype=float)
    result["vs_h1"] = _metrics(pred1, h1)

    if COL_H2 in df.columns:
        sub2 = df[[COL_PRED, COL_H2]].dropna()
        pred2 = sub2[COL_PRED].to_numpy(dtype=float)
        h2    = sub2[COL_H2].to_numpy(dtype=float)
        result["vs_h2"] = _metrics(pred2, h2)

        # Inter-rater: align on rows where both labels are present
        sub_ir = df[[COL_H1, COL_H2]].dropna()
        ir_h1 = sub_ir[COL_H1].to_numpy(dtype=float)
        ir_h2 = sub_ir[COL_H2].to_numpy(dtype=float)
        result["inter_rater"] = _metrics(ir_h1, ir_h2)

        # Average-human baseline: compare model against mean of the two labelers
        sub_avg = df[[COL_PRED, COL_H1, COL_H2]].dropna()
        pred_avg = sub_avg[COL_PRED].to_numpy(dtype=float)
        avg_human = ((sub_avg[COL_H1] + sub_avg[COL_H2]) / 2).to_numpy(dtype=float)
        result["vs_avg"] = _metrics(pred_avg, avg_human)

    return result


def _print_metrics_block(m: dict, label_a: str, label_b: str, indent: str = "  ") -> None:
    print(f"{indent}  Pairs evaluated: {m['n']}")
    print(f"{indent}  Score distributions:")
    print(f"{indent}    {'':30s} {'Mean':>8} {'Std':>8}")
    print(f"{indent}    {'-'*47}")
    print(f"{indent}    {label_a:30s} {m['a_mean']:>8.4f} {m['a_std']:>8.4f}")
    print(f"{indent}    {label_b:30s} {m['b_mean']:>8.4f} {m['b_std']:>8.4f}")
    print(f"{indent}  Metrics:")
    print(f"{indent}    MAE         = {m['mae']:.4f}  (mean |a - b| over {m['n']} pairs)")
    print(f"{indent}    RMSE        = {m['rmse']:.4f}  (sqrt(mean((a-b)^2)); penalizes outliers)")
    sig = "  *significant*" if m["pearson_p"] < 0.05 else "  not significant"
    print(f"{indent}    Pearson r   = {m['pearson_r']:.4f}  (p = {m['pearson_p']:.4f}{sig})")
    print(f"{indent}    L2 distance = {m['l2_distance']:.4f}  (sqrt(sum((a-b)^2)))")


def print_relevance_results(csv_path) -> None:
    """Load the results CSV, filter to strategy D, and print relevance metrics."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    d_df = df[df["strategy"] == "D"].copy() if "strategy" in df.columns else df.copy()
    if d_df.empty:
        print("No strategy D rows found — skipping relevance evaluation.")
        return

    if COL_H1 not in d_df.columns:
        print(f"Column '{COL_H1}' not found — skipping relevance evaluation.")
        return

    has_h2 = COL_H2 in d_df.columns and d_df[COL_H2].notna().any()

    print_section("Strategy D — Relevance Score Evaluation")
    all_m = compute_relevance_metrics(d_df)

    print("\n--- Model vs. Human Labeler 1 ---")
    _print_metrics_block(all_m["vs_h1"], COL_PRED, COL_H1)

    if has_h2:
        print("\n--- Model vs. Human Labeler 2 ---")
        _print_metrics_block(all_m["vs_h2"], COL_PRED, COL_H2)

        print("\n--- Model vs. Average of Both Labelers ---")
        _print_metrics_block(all_m["vs_avg"], COL_PRED, "avg(H1, H2)")

        print("\n--- Inter-Rater Agreement (Human 1 vs. Human 2) ---")
        _print_metrics_block(all_m["inter_rater"], COL_H1, COL_H2)

    print()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(RESULTS_CSV)
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    print_relevance_results(path)
