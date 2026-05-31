"""
Evaluation utilities: scoring model predictions against ground-truth labels.

Ground-truth label meaning (column "support"):
  0 = Attack  |  1 = Support  |  2 = No Relation

Prompt strategies:
  A — binary:      only predicts attack            (pred_attack)
  B — two-class:   predicts attack or support      (pred_attack, pred_support)
  C — three-class: predicts attack, support, neither (pred_attack, pred_support, pred_neither)
"""

import pandas as pd

SUPPORT_MAP = {0: "Attack", 1: "Support", 2: "No Relation"}


def compute_correct(row) -> int | pd.NA:
    """Return 1 if the model prediction matches the ground truth, 0 if not, NA if prediction is missing.

    Strategy A — binary (pred_attack only):
      Correct when support==0 (Attack) ↔ pred_attack==1, otherwise pred_attack==0.

    Strategy B — two-class (pred_attack, pred_support):
      support==0 → pred_attack must be 1
      support==1 → pred_support must be 1
      support==2 → always 0 (no "neither" class in B)

    Strategy C — three-class (pred_attack, pred_support, pred_neither):
      support==0 → pred_attack must be 1
      support==1 → pred_support must be 1
      support==2 → pred_neither must be 1
    """
    s = row["support"]
    strat = row["strategy"]

    if strat == "A":
        if pd.isna(row["pred_attack"]):
            return pd.NA
        return int((s == 0) == (row["pred_attack"] == 1))

    elif strat == "B":
        if pd.isna(row["pred_attack"]) or pd.isna(row["pred_support"]):
            return pd.NA
        if s == 0:
            return int(row["pred_attack"] == 1)
        elif s == 1:
            return int(row["pred_support"] == 1)
        else:  # support==2, no "neither" class in B
            return 0

    elif strat == "C":
        if pd.isna(row["pred_attack"]) or pd.isna(row["pred_support"]) or pd.isna(row["pred_neither"]):
            return pd.NA
        if s == 0:
            return int(row["pred_attack"] == 1)
        elif s == 1:
            return int(row["pred_support"] == 1)
        else:
            return int(row["pred_neither"] == 1)

    return pd.NA


def add_correct_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'correct' column to a predictions dataframe (modifies in place, returns df)."""
    df["correct"] = df.apply(compute_correct, axis=1).astype("Int64")
    return df


def accuracy_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return per-strategy accuracy as a dict {strategy: accuracy_fraction}."""
    summary = {}
    for strat, grp in df.groupby("strategy"):
        valid = grp["correct"].dropna()
        if len(valid):
            summary[strat] = valid.mean()
    return summary
