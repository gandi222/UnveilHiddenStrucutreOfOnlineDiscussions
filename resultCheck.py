"""
resultCheck.py — compute F1 score from results2.csv, printing every step.

Ground-truth label (column "support [true value]"):
  0 = Attack  |  1 = Support  |  2 = No Relation

Strategy A — binary: model predicts attack=0/1 only.
  Collapsed to binary: true positive class = Attack (support==0),
  negative class = Not-Attack (support==1 or 2).

Strategy B — two-class: model predicts attack=0/1 and support=0/1.
  Predicted label = Attack if pred_attack==1, else Support if pred_support==1, else No-Relation.
  Macro F1 over all 3 ground-truth classes.

Strategy C — three-class: model predicts attack, support, neither.
  Predicted label = Attack / Support / No-Relation based on which flag == 1.
  Macro F1 over all 3 classes.
"""

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "results2.csv"

LABEL_MAP = {0: "Attack", 1: "Support", 2: "No Relation"}


# ── label derivation ──────────────────────────────────────────────────────────

def derive_labels_a(df: pd.DataFrame):
    """Binary: Attack vs Not-Attack."""
    valid = df.dropna(subset=["pred_attack"])
    y_true = ["Attack" if s == 0 else "Not-Attack"
              for s in valid["support [true value]"]]
    y_pred = ["Attack" if p == 1 else "Not-Attack"
              for p in valid["pred_attack"]]
    classes = ["Attack", "Not-Attack"]
    return y_true, y_pred, classes, len(df) - len(valid)


def derive_labels_b(df: pd.DataFrame):
    """Two-class: Attack or Support predicted; No Relation never predicted."""
    valid = df.dropna(subset=["pred_attack", "pred_support"])
    y_true = [LABEL_MAP[s] for s in valid["support [true value]"]]
    y_pred = []
    for _, row in valid.iterrows():
        if row["pred_attack"] == 1:
            y_pred.append("Attack")
        elif row["pred_support"] == 1:
            y_pred.append("Support")
        else:
            y_pred.append("No Relation")  # model output both 0 (invalid per prompt)
    classes = ["Attack", "Support", "No Relation"]
    return y_true, y_pred, classes, len(df) - len(valid)


def derive_labels_c(df: pd.DataFrame):
    """Three-class: Attack, Support, or No Relation."""
    valid = df.dropna(subset=["pred_attack", "pred_support", "pred_neither"])
    y_true = [LABEL_MAP[s] for s in valid["support [true value]"]]
    y_pred = []
    for _, row in valid.iterrows():
        if row["pred_attack"] == 1:
            y_pred.append("Attack")
        elif row["pred_support"] == 1:
            y_pred.append("Support")
        elif row["pred_neither"] == 1:
            y_pred.append("No Relation")
        else:
            y_pred.append(None)  # all flags == 0 (invalid per prompt)
    classes = ["Attack", "Support", "No Relation"]
    return y_true, y_pred, classes, len(df) - len(valid)


# ── F1 computation ────────────────────────────────────────────────────────────

def compute_f1(y_true, y_pred, classes):
    """Return per-class metrics dict and macro F1. Skips None predictions."""
    # filter out None predictions
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    n_skipped = len(y_true) - len(pairs)

    metrics = {}
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        tn = sum(1 for t, p in pairs if t != cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics[cls] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1,
        }

    macro_f1 = sum(m["F1"] for m in metrics.values()) / len(classes)
    return metrics, macro_f1, n_skipped


# ── pretty printing ───────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_strategy(strategy: str, df: pd.DataFrame):
    print_section(f"Strategy {strategy}")

    derive_fn = {"A": derive_labels_a, "B": derive_labels_b, "C": derive_labels_c}[strategy]
    y_true, y_pred, classes, n_missing_preds = derive_fn(df)

    total = len(y_true)
    print(f"\nTotal rows for strategy {strategy}: {total}")
    if n_missing_preds:
        print(f"  Rows skipped (missing predictions): {n_missing_preds}")

    # ── ground-truth distribution ──────────────────────────────────────────
    print("\nGround-truth label distribution:")
    gt_counts = pd.Series(y_true).value_counts()
    for cls in classes:
        print(f"  {cls:12s}: {gt_counts.get(cls, 0):4d}")

    # ── prediction distribution ────────────────────────────────────────────
    print("\nPredicted label distribution:")
    pred_counts = pd.Series([p for p in y_pred if p is not None]).value_counts()
    for cls in classes:
        print(f"  {cls:12s}: {pred_counts.get(cls, 0):4d}")

    # ── confusion matrix ───────────────────────────────────────────────────
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = f"  {'':14s}" + "".join(f"{c:>14s}" for c in classes)
    print(header)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    for true_cls in classes:
        row_vals = [sum(1 for t, p in pairs if t == true_cls and p == pred_cls)
                    for pred_cls in classes]
        row_str = "".join(f"{v:>14d}" for v in row_vals)
        print(f"  {true_cls:14s}{row_str}")

    # ── per-class breakdown ────────────────────────────────────────────────
    metrics, macro_f1, n_skipped = compute_f1(y_true, y_pred, classes)

    print("\nPer-class breakdown  (F1 = 2·TP / (2·TP + FP + FN)):")
    print(f"  {'Class':14s} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*72}")
    for cls in classes:
        m = metrics[cls]
        denom_check = 2 * m["TP"] + m["FP"] + m["FN"]
        print(f"  {cls:14s} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
              f"{m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f}"
              f"  [2·{m['TP']}/(2·{m['TP']}+{m['FP']}+{m['FN']}) = "
              f"2·{m['TP']}/{denom_check}]")

    # ── macro F1 ───────────────────────────────────────────────────────────
    f1_values = [f"{metrics[c]['F1']:.4f}" for c in classes]
    print(f"\nMacro F1 = average of per-class F1 scores")
    print(f"  = ({' + '.join(f1_values)}) / {len(classes)}")
    print(f"  = {macro_f1:.4f}")

    # ── accuracy (cross-check with main.py) ───────────────────────────────
    correct = sum(1 for t, p in pairs if t == p)
    accuracy = correct / len(pairs) if pairs else 0.0
    print(f"\nAccuracy (cross-check): {correct}/{len(pairs)} = {accuracy:.4f}"
          f"  ({100*accuracy:.1f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
    print(f"Strategies present: {sorted(df['strategy'].unique())}")

    for strategy in sorted(df["strategy"].unique()):
        if strategy not in ("A", "B", "C"):
            print(f"  Skipping unknown strategy '{strategy}'")
            continue
        print_strategy(strategy, df[df["strategy"] == strategy].copy())

    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")


if __name__ == "__main__":
    main()
