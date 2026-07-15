"""
human_label_evaluation.py — evaluate human-labeled support classifications.

Compares the human annotator's labels ("support human labeled") against the
original dataset ground truth ("support") using accuracy, per-class F1,
macro F1, and a confusion matrix.

Run standalone with:  python human_label_evaluation.py [path/to/file.csv]
Default file:         QBAF_relevanceScoreHumanLabeled.csv
"""

import sys
from pathlib import Path

import pandas as pd

from evaluation import SUPPORT_MAP, compute_f1, print_section

INPUT_CSV  = "QBAF_relevanceScoreHumanLabeled.csv"
COL_TRUE   = "support"
COL_PRED   = "support human labeled"
CLASSES    = ["Attack", "Support", "No Relation"]


def print_human_label_results(csv_path) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    sub = df[[COL_TRUE, COL_PRED]].dropna()
    n_dropped = len(df) - len(sub)

    y_true = [SUPPORT_MAP[int(v)] for v in sub[COL_TRUE]]
    y_pred = [SUPPORT_MAP[int(v)] for v in sub[COL_PRED]]
    n = len(y_true)

    print_section("Human Label Evaluation — Support Classification")
    print(f"\n  Comparing: '{COL_PRED}'  vs.  ground truth '{COL_TRUE}'")
    print(f"  File:      {csv_path.name}")
    print(f"  Pairs evaluated: {n}" + (f"  ({n_dropped} dropped — missing values)" if n_dropped else ""))

    # ── label distributions ────────────────────────────────────────────────────
    true_counts = pd.Series(y_true).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\n  {'Label':14s}  {'Ground truth':>12}  {'Human label':>11}")
    print(f"  {'-'*42}")
    for cls in CLASSES:
        print(f"  {cls:14s}  {true_counts.get(cls, 0):>12}  {pred_counts.get(cls, 0):>11}")

    # ── confusion matrix ───────────────────────────────────────────────────────
    print("\n  Confusion matrix (rows = ground truth, cols = human label):")
    pairs = list(zip(y_true, y_pred))
    header = f"  {'':16s}" + "".join(f"{c:>14s}" for c in CLASSES)
    print(header)
    for true_cls in CLASSES:
        row_vals = [sum(1 for t, p in pairs if t == true_cls and p == pred_cls)
                    for pred_cls in CLASSES]
        print(f"  {true_cls:16s}" + "".join(f"{v:>14d}" for v in row_vals))

    # ── per-class F1 ──────────────────────────────────────────────────────────
    metrics, macro_f1, _ = compute_f1(y_true, y_pred, CLASSES)

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':14s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-'*47}")
    for cls in CLASSES:
        m = metrics[cls]
        print(f"  {cls:14s}  {m['Precision']:>10.4f}  {m['Recall']:>8.4f}  {m['F1']:>8.4f}")

    # ── macro F1 + accuracy ────────────────────────────────────────────────────
    f1_vals = " + ".join(f"{metrics[c]['F1']:.4f}" for c in CLASSES)
    print(f"\n  Macro F1 = ({f1_vals}) / {len(CLASSES)} = {macro_f1:.4f}")

    n_correct = sum(t == p for t, p in pairs)
    accuracy = n_correct / n if n else 0.0
    print(f"  Accuracy = {n_correct}/{n} = {accuracy:.4f}  ({100 * accuracy:.2f}%)")
    print()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(INPUT_CSV)
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    print_human_label_results(path)
