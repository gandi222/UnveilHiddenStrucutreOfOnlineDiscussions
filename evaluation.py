"""
evaluation.py — standalone evaluation of results2.csv.

Run with:  python evaluation.py [path/to/results2.csv]

Ground-truth label (column "support [true value]"):
  0 = Attack  |  1 = Support  |  2 = No Relation

Strategy A — binary: model predicts attack=0/1 only.
  Correct when (support==0) ↔ (pred_attack==1).
  Collapsed to binary for F1: positive class = Attack, negative = Not-Attack.

Strategy B — two-class: model predicts attack=0/1 and support=0/1.
  Correct when support==0→pred_attack==1, support==1→pred_support==1; support==2 always wrong.
  Macro F1 over all 3 ground-truth classes.

Strategy C — three-class: model predicts attack, support, neither.
  Correct when the matching flag == 1.
  Macro F1 over all 3 classes.

Strategy D — three-class + relevance score: evaluated identically to C.
  The relevance score (pred_relevance) is ignored for correctness / F1 purposes.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Exported so pipeline.py can import it for the support_label column.
SUPPORT_MAP = {0: "Attack", 1: "Support", 2: "No Relation"}

CSV_PATH = Path(__file__).parent / "results2.csv"


# ── accuracy ──────────────────────────────────────────────────────────────────

def compute_correct_row(row):
    """Return 1/0/None for one row from results2.csv.

    Strategy A: correct ↔ (support==0) == (pred_attack==1)
    Strategy B: support==0→pred_attack==1, support==1→pred_support==1, support==2→always 0
    Strategy C: support==0→pred_attack==1, support==1→pred_support==1, support==2→pred_neither==1
    Returns None when a required prediction column is missing.
    """
    s = row["support [true value]"]
    strat = row["strategy"]

    if strat == "A":
        if pd.isna(row["pred_attack"]):
            return None
        return int((s == 0) == (row["pred_attack"] == 1))

    elif strat == "B":
        if pd.isna(row["pred_attack"]) or pd.isna(row["pred_support"]):
            return None
        if s == 0:
            return int(row["pred_attack"] == 1)
        elif s == 1:
            return int(row["pred_support"] == 1)
        else:
            return 0  # No Relation is never correct in B

    elif strat in ("C", "D"):
        if pd.isna(row["pred_attack"]) or pd.isna(row["pred_support"]) or pd.isna(row["pred_neither"]):
            return None
        if s == 0:
            return int(row["pred_attack"] == 1)
        elif s == 1:
            return int(row["pred_support"] == 1)
        else:
            return int(row["pred_neither"] == 1)

    return None


def accuracy_for_strategy(df: pd.DataFrame):
    """Return (correct_count, valid_count) for a single-strategy dataframe."""
    results = [compute_correct_row(row) for _, row in df.iterrows()]
    valid = [r for r in results if r is not None]
    return sum(valid), len(valid)


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
    y_true = [SUPPORT_MAP[s] for s in valid["support [true value]"]]
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
    y_true = [SUPPORT_MAP[s] for s in valid["support [true value]"]]
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
    """Return per-class metrics dict, macro F1, and count of skipped (None) predictions."""
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

    derive_fn = {"A": derive_labels_a, "B": derive_labels_b, "C": derive_labels_c, "D": derive_labels_c}[strategy]
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
    print("\nConfusion matrix (rows = true label, cols = predicted label):")
    print("  Each cell [row R, col C] = number of samples whose TRUE label is R")
    print("  and whose PREDICTED label is C.")
    print("  Diagonal = correct predictions. Off-diagonal = errors.")
    header = f"  {'':14s}" + "".join(f"{c:>14s}" for c in classes)
    print(header)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    for true_cls in classes:
        row_vals = [sum(1 for t, p in pairs if t == true_cls and p == pred_cls)
                    for pred_cls in classes]
        row_str = "".join(f"{v:>14d}" for v in row_vals)
        print(f"  {true_cls:14s}{row_str}")

    # ── per-class breakdown ────────────────────────────────────────────────
    metrics, macro_f1, _ = compute_f1(y_true, y_pred, classes)
    n_valid_pairs = len(pairs)

    print("\nDefinitions (per class C, treating C as the positive class):")
    print("  TP = true  label == C  AND  predicted label == C  (correct positive)")
    print("  FP = true  label != C  AND  predicted label == C  (false alarm)")
    print("  FN = true  label == C  AND  predicted label != C  (missed positive)")
    print("  TN = true  label != C  AND  predicted label != C  (correct negative)")
    print("  Double-check: TP + FP + FN + TN must equal total valid pairs for every class.")

    print(f"\nPer-class breakdown  (F1 = 2·TP / (2·TP + FP + FN)):")
    print(f"  {'Class':14s} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}  {'Sum check':>20}")
    print(f"  {'-'*95}")
    all_sums_ok = True
    for cls in classes:
        m = metrics[cls]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        prec_denom = tp + fp
        rec_denom  = tp + fn
        f1_denom   = 2 * tp + fp + fn
        total_check = tp + fp + fn + m["TN"]
        ok = total_check == n_valid_pairs
        if not ok:
            all_sums_ok = False
        check_str = f"[{total_check}=={n_valid_pairs} {'OK' if ok else 'FAIL'}]"
        print(f"  {cls:14s} {tp:>5} {fp:>5} {fn:>5} {m['TN']:>5}  "
              f"{m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f}"
              f"  [2·{tp}/{f1_denom}]  {check_str}")
        print(f"    Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {tp}/{prec_denom} = {m['Precision']:.4f}")
        print(f"    Recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {tp}/{rec_denom} = {m['Recall']:.4f}")
    if all_sums_ok:
        print(f"  => All class sums equal total valid pairs ({n_valid_pairs}). OK.")

    # ── macro F1 ───────────────────────────────────────────────────────────
    f1_values = [f"{metrics[c]['F1']:.4f}" for c in classes]
    print(f"\nMacro F1 = average of per-class F1 scores")
    print(f"  = ({' + '.join(f1_values)}) / {len(classes)}")
    print(f"  = {macro_f1:.4f}")

    # ── accuracy ───────────────────────────────────────────────────────────
    n_correct, n_valid = accuracy_for_strategy(df)
    acc = n_correct / n_valid if n_valid else 0.0
    print(f"\nAccuracy: {n_correct}/{n_valid} correct = {acc:.4f}  ({100*acc:.4f}%)"
          + (f"  [{len(df) - n_valid} rows skipped — missing predictions]"
             if len(df) - n_valid else ""))


# ── overview helpers ──────────────────────────────────────────────────────────

_DERIVE_FNS = {"A": "derive_labels_a", "B": "derive_labels_b",
               "C": "derive_labels_c", "D": "derive_labels_c"}


def compute_metrics(df: pd.DataFrame, strategy: str) -> dict:
    """Return a flat metrics dict for a single-strategy dataframe.

    Keys: n_evaluated, n_correct, accuracy, macro_f1,
          prec/rec/f1 for attack, support, no_relation (NaN when not applicable).
    """
    derive_fn = {"A": derive_labels_a, "B": derive_labels_b,
                 "C": derive_labels_c, "D": derive_labels_c}[strategy]
    y_true, y_pred, classes, _ = derive_fn(df)
    per_class, macro_f1, _ = compute_f1(y_true, y_pred, classes)
    n_correct, n_valid = accuracy_for_strategy(df)

    result = {
        "n_evaluated": len(df),
        "n_correct": n_correct,
        "accuracy": n_correct / n_valid if n_valid else float("nan"),
        "macro_f1": macro_f1,
    }
    for label, key in [("Attack", "attack"), ("Support", "support"), ("No Relation", "no_relation")]:
        if label in per_class:
            m = per_class[label]
            result[f"true_positive_{key}"]  = m["TP"]
            result[f"false_positive_{key}"] = m["FP"]
            result[f"false_negative_{key}"] = m["FN"]
            result[f"true_negative_{key}"]  = m["TN"]
            result[f"precision_{key}"]      = m["Precision"]
            result[f"recall_{key}"]         = m["Recall"]
            result[f"f1_score_{key}"]       = m["F1"]
        else:
            for pfx in ("true_positive", "false_positive", "false_negative", "true_negative",
                        "precision", "recall", "f1_score"):
                result[f"{pfx}_{key}"] = float("nan")
    return result


def append_to_overview(results_csv: str, strategy: str, config: dict,
                       overview_csv: str = "ResultOverview_allTests.csv") -> None:
    """Compute metrics for *strategy* in results_csv and append one row to overview_csv."""


    df = pd.read_csv(results_csv)
    strat_df = df[df["strategy"] == strategy].copy()
    if strat_df.empty:
        return

    metrics = compute_metrics(strat_df, strategy)

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    counts = strat_df["support [true value]"].value_counts()
    few_shot_n = config.get("few_shot_n", 0)

    pred_attack      = int(strat_df["pred_attack"].fillna(0).sum())
    pred_support     = int(strat_df["pred_support"].fillna(0).sum())
    pred_no_relation = int(strat_df["pred_neither"].fillna(0).sum())

    row = {
        # ── Run metadata ──────────────────────────────────────────────────
        "timestamp":                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit":                     git_hash,
        "model":                          config.get("model"),
        "strategy":                       strategy,
        "few_shot_n":                     few_shot_n,
        "dataset_file":                   config.get("dataset_file"),
        "limit_config":                   config.get("limit"),
        "batch_size":                     config.get("batch_size"),
        "delay_seconds":                  config.get("delay_seconds"),
        "max_retries":                    config.get("max_retries"),
        # ── Dataset structure ─────────────────────────────────────────────
        "n_evaluated":                    metrics["n_evaluated"],
        "ground_truth_count_attack":      int(counts.get(0, 0)),
        "ground_truth_count_support":     int(counts.get(1, 0)),
        "ground_truth_count_no_relation": int(counts.get(2, 0)),
        "predicted_count_attack":         pred_attack,
        "predicted_count_support":        pred_support,
        "predicted_count_no_relation":    pred_no_relation,
        # ── Aggregate metrics ─────────────────────────────────────────────
        "accuracy":                       metrics["accuracy"],
        # ── Per-class confusion matrix ────────────────────────────────────
        "true_positive_attack":           metrics["true_positive_attack"],
        "false_positive_attack":          metrics["false_positive_attack"],
        "false_negative_attack":          metrics["false_negative_attack"],
        "true_negative_attack":           metrics["true_negative_attack"],
        "true_positive_support":          metrics["true_positive_support"],
        "false_positive_support":         metrics["false_positive_support"],
        "false_negative_support":         metrics["false_negative_support"],
        "true_negative_support":          metrics["true_negative_support"],
        "true_positive_no_relation":      metrics["true_positive_no_relation"],
        "false_positive_no_relation":     metrics["false_positive_no_relation"],
        "false_negative_no_relation":     metrics["false_negative_no_relation"],
        "true_negative_no_relation":      metrics["true_negative_no_relation"],
        # ── Per-class precision ───────────────────────────────────────────
        "precision_attack":               metrics["precision_attack"],
        "precision_support":              metrics["precision_support"],
        "precision_no_relation":          metrics["precision_no_relation"],
        # ── Per-class recall ──────────────────────────────────────────────
        "recall_attack":                  metrics["recall_attack"],
        "recall_support":                 metrics["recall_support"],
        "recall_no_relation":             metrics["recall_no_relation"],
        # ── Per-class F1 ──────────────────────────────────────────────────
        "f1_score_attack":                metrics["f1_score_attack"],
        "f1_score_support":               metrics["f1_score_support"],
        "f1_score_no_relation":           metrics["f1_score_no_relation"],
        # ── Macro F1 (at the end) ─────────────────────────────────────────
        "macro_f1":                       metrics["macro_f1"],
    }

    overview_path = Path(overview_csv)
    if overview_path.exists():
        existing = pd.read_csv(overview_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(overview_path, index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def print_all_results(csv_path) -> None:
    """Print per-strategy metrics and accuracy summary for the given results CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Strategies present: {sorted(df['strategy'].unique())}")

    strategies = [s for s in sorted(df["strategy"].unique()) if s in ("A", "B", "C", "D")]
    for strategy in strategies:
        print_strategy(strategy, df[df["strategy"] == strategy].copy())

    print_section("Accuracy summary")
    for strategy in strategies:
        n_correct, n_valid = accuracy_for_strategy(df[df["strategy"] == strategy])
        acc = n_correct / n_valid if n_valid else 0.0
        print(f"  Strategy {strategy}: {n_correct}/{n_valid} correct ({100*acc:.4f}%)")

    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)
    print_all_results(csv_path)


if __name__ == "__main__":
    main()
