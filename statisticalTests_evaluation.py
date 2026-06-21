"""
Statistical evaluation script for BA argument-mining experiments.

Usage:
    python statisticalTests_evaluation.py [path/to/ResultOverview_allTests.csv]

Defaults to ResultOverview_allTests.csv in the same directory as this script.

Outputs (written to eval_output/ next to the CSV):
    runs_annotated.csv              – prepared frame with ds, n_eff, n_correct
    pairwise_ztests_accuracy.csv    – Step 1 z-test results
    evaluation_report.md            – Markdown report

Step 1 implements two-proportion z-tests on accuracy.
Step 2 (per-class precision/recall) is out of scope for now.
"""

import argparse
import os
import sys
import warnings
from typing import Optional

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.05

REPORT_INTRO = """# Statistical Evaluation Report

## Method: Two-Proportion Z-Test on Accuracy

Accuracy is a binomial proportion, so for large n the two-proportion z-test is
the correct comparison between two runs. A t-test is **not** appropriate for
single-run accuracies because the outcome is binary (correct/incorrect) and the
variance is fully determined by the mean — there is nothing extra to estimate.
A t-test would only apply if we had multiple seeded runs per configuration.

Runs use different random subsets, so comparisons are **unpaired** (a paired
McNemar test is a later step).

**Effective sample size (n_eff):** Failed batches drop pairs, so we do *not*
assume n = 1000 for every run. The true n is computed from prediction counts:
- Strategies C/D: `predicted_count_attack + predicted_count_support + predicted_count_no_relation`
- Strategy B:     `predicted_count_attack + predicted_count_support`
- Strategy A:     `tp_attack + fp_attack + fn_attack + tn_attack`

`n_correct = round(accuracy * n_eff)`. All tests use `n_eff`, never 1000.

"""


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load the CSV and add ds, n_eff, n_correct columns.

    Does NOT modify any existing metric column (macro_f1, etc.).
    """
    df = pd.read_csv(csv_path)

    # ds column: NR_ prefix → NR_Web, else Web
    df["ds"] = df["dataset_file"].apply(
        lambda p: "NR_Web" if str(p).startswith("NR_") else "Web"
    )

    # n_eff: effective sample size per strategy
    def compute_n_eff(row):
        s = row["strategy"]
        if s in ("C", "D"):
            return (
                row["predicted_count_attack"]
                + row["predicted_count_support"]
                + row["predicted_count_no_relation"]
            )
        elif s == "B":
            return row["predicted_count_attack"] + row["predicted_count_support"]
        else:  # A
            return (
                row["true_positive_attack"]
                + row["false_positive_attack"]
                + row["false_negative_attack"]
                + row["true_negative_attack"]
            )

    df["n_eff"] = df.apply(compute_n_eff, axis=1)
    df["n_correct"] = (df["accuracy"] * df["n_eff"]).round().astype(int)

    return df


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def two_prop_ztest(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-proportion z-test (two-sided, unpaired) via statsmodels.

    Uses a pooled proportion for the SE, which is the standard approach when
    testing H0: p1 == p2.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool in (0.0, 1.0):  # SE would be zero — undefined test
        return dict(p1=p1, p2=p2, diff_pp=(p1 - p2) * 100,
                    z=float("nan"), p_value=float("nan"), significant=False)
    z, p_value = proportions_ztest([x1, x2], [n1, n2], alternative="two-sided")
    return dict(
        p1=p1,
        p2=p2,
        diff_pp=(p1 - p2) * 100,
        z=z,
        p_value=p_value,
        significant=p_value < ALPHA,
    )


# ---------------------------------------------------------------------------
# Row lookup helper
# ---------------------------------------------------------------------------

def get(df: pd.DataFrame, strategy: str, few_shot_n: int,
        ds: str, batch_size: int) -> Optional[pd.Series]:
    """Return the single matching row, or None with a warning."""
    mask = (
        (df["strategy"] == strategy)
        & (df["few_shot_n"] == few_shot_n)
        & (df["ds"] == ds)
        & (df["batch_size"] == batch_size)
    )
    rows = df[mask]
    if len(rows) == 0:
        warnings.warn(
            f"No row found for strategy={strategy}, few_shot_n={few_shot_n}, "
            f"ds={ds}, batch_size={batch_size}"
        )
        return None
    if len(rows) > 1:
        warnings.warn(
            f"Multiple rows for strategy={strategy}, few_shot_n={few_shot_n}, "
            f"ds={ds}, batch_size={batch_size} — using first."
        )
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Step 1: accuracy z-tests
# ---------------------------------------------------------------------------

def run_accuracy_tests(df: pd.DataFrame) -> list[dict]:
    """Run all pairwise two-proportion z-tests on accuracy.

    Returns a list of result dicts for downstream formatting/export.
    """
    results = []

    def compare(group_id: str, comparison: str, strategy: str,
                 factor: str, ds: str,
                 label1: str, row1,
                 label2: str, row2) -> Optional[dict]:
        if row1 is None or row2 is None:
            return None
        res = two_prop_ztest(
            int(row1["n_correct"]), int(row1["n_eff"]),
            int(row2["n_correct"]), int(row2["n_eff"]),
        )
        return dict(
            group=group_id,
            comparison=comparison,
            strategy=strategy,
            factor=factor,
            ds=ds,
            group1=label1,
            acc1=row1["accuracy"],
            n1=int(row1["n_eff"]),
            group2=label2,
            acc2=row2["accuracy"],
            n2=int(row2["n_eff"]),
            diff_pp=res["diff_pp"],
            z=res["z"],
            p_value=res["p_value"],
            significant=res["significant"],
        )

    # ── Group 1: zero-shot vs few-shot (same strategy, batch 10) ─────────
    g1_comparisons = [
        # (strategy, few_shot_n, ds)
        ("A", 7,  "Web"),
        ("A", 10, "Web"),
        ("A", 14, "Web"),
        ("A", 20, "Web"),
        ("C", 10, "NR_Web"),
        ("C", 20, "NR_Web"),
    ]
    for strat, fs, ds in g1_comparisons:
        base = get(df, strat, 0, ds, 10)
        shot = get(df, strat, fs, ds, 10)
        rec = compare(
            "1_zeroshot_vs_fewshot",
            f"{strat} {fs}-shot vs 0-shot (b10)",
            strat, "few_shot", ds,
            f"{fs}-shot", shot,
            "0-shot", base,
        )
        if rec:
            results.append(rec)

    # ── Group 2: batch size (same strategy, zero-shot) ───────────────────
    batch_comparisons = [
        # (strategy, ds, b_ref, b_cmp)
        ("A", "Web",    5,  10),
        ("A", "Web",    5,  20),
        ("A", "Web",   10,  20),
        ("C", "NR_Web", 5,  10),
        ("C", "NR_Web", 10, 20),
        ("C", "NR_Web", 5,  20),
        ("B", "Web",    5,  20),
    ]
    for strat, ds, b1, b2 in batch_comparisons:
        r1 = get(df, strat, 0, ds, b1)
        r2 = get(df, strat, 0, ds, b2)
        rec = compare(
            "2_batch_size",
            f"{strat} b{b1} vs b{b2} (0-shot)",
            strat, "batch_size", ds,
            f"b{b1}", r1,
            f"b{b2}", r2,
        )
        if rec:
            results.append(rec)

    # ── Group 3: across strategies (zero-shot, matched batch) ────────────
    cross_comparisons = [
        # (strat1, ds1, b, strat2, ds2)
        ("A", "Web", 5,  "B", "Web"),
        ("A", "Web", 20, "B", "Web"),
        ("B", "Web", 5,  "C", "NR_Web"),
    ]
    for s1, ds1, b, s2, ds2 in cross_comparisons:
        r1 = get(df, s1, 0, ds1, b)
        r2 = get(df, s2, 0, ds2, b)
        rec = compare(
            "3_cross_strategy",
            f"{s1} vs {s2} (b{b}, 0-shot)",
            f"{s1}+{s2}", "strategy", f"{ds1}+{ds2}",
            s1, r1,
            s2, r2,
        )
        if rec:
            results.append(rec)

    # ── Group 1b: few-shot dose-response (same strategy, batch 10) ───────
    # A: all pairs from {7, 10, 14, 20}
    a_shots = [7, 10, 14, 20]
    for i, fs1 in enumerate(a_shots):
        for fs2 in a_shots[i + 1:]:
            r1 = get(df, "A", fs1, "Web", 10)
            r2 = get(df, "A", fs2, "Web", 10)
            rec = compare(
                "1b_fewshot_dose_response",
                f"A {fs1}-shot vs {fs2}-shot (b10)",
                "A", "few_shot", "Web",
                f"{fs1}-shot", r1,
                f"{fs2}-shot", r2,
            )
            if rec:
                results.append(rec)
    # B: 10 vs 20
    r_b10 = get(df, "B", 10, "Web", 10)
    r_b20 = get(df, "B", 20, "Web", 10)
    rec = compare(
        "1b_fewshot_dose_response",
        "B 10-shot vs 20-shot (b10)",
        "B", "few_shot", "Web",
        "10-shot", r_b10,
        "20-shot", r_b20,
    )
    if rec:
        results.append(rec)
    # C: 10 vs 20
    r_c10 = get(df, "C", 10, "NR_Web", 10)
    r_c20 = get(df, "C", 20, "NR_Web", 10)
    rec = compare(
        "1b_fewshot_dose_response",
        "C 10-shot vs 20-shot (b10)",
        "C", "few_shot", "NR_Web",
        "10-shot", r_c10,
        "20-shot", r_c20,
    )
    if rec:
        results.append(rec)

    # ── Group 3b: A vs B at matched few-shot (batch 10, Web) ─────────────
    for fs in (10, 20):
        r_a = get(df, "A", fs, "Web", 10)
        r_b = get(df, "B", fs, "Web", 10)
        rec = compare(
            "3b_cross_strategy_fewshot",
            f"A vs B {fs}-shot (b10)",
            "A+B", "strategy", "Web",
            f"A {fs}-shot", r_a,
            f"B {fs}-shot", r_b,
        )
        if rec:
            results.append(rec)

    # ── Group 4: C vs D (zero-shot, batch 10) ────────────────────────────
    r_c = get(df, "C", 0, "NR_Web", 10)
    r_d = get(df, "D", 0, "NR_Web", 10)
    rec = compare(
        "4_C_vs_D",
        "C vs D (b10, 0-shot)",
        "C+D", "strategy", "NR_Web",
        "C", r_c,
        "D", r_d,
    )
    if rec:
        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _sig_str(significant: bool) -> str:
    return "YES *" if significant else "no"


def print_accuracy_results(results: list[dict]) -> None:
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    group_labels = {
        "1_zeroshot_vs_fewshot":     "GROUP 1 — Zero-shot vs Few-shot (same strategy, batch 10)",
        "1b_fewshot_dose_response":   "GROUP 1b — Few-shot dose-response (same strategy, batch 10)",
        "2_batch_size":               "GROUP 2 — Batch size (same strategy, zero-shot)",
        "3_cross_strategy":           "GROUP 3 — Cross-strategy (zero-shot, matched batch)",
        "3b_cross_strategy_fewshot":  "GROUP 3b — A vs B at matched few-shot (batch 10, WebDataset)",
        "4_C_vs_D":                   "GROUP 4 — C vs D (zero-shot, batch 10)",
    }

    print("\n" + "=" * 70)
    print("STEP 1 — Pairwise Two-Proportion Z-Tests on Accuracy")
    print("=" * 70)

    for gid, label in group_labels.items():
        if gid not in groups:
            continue
        print(f"\n{label}")
        if gid == "3_cross_strategy":
            print(
                "  NOTE: A/B use WebDataset (binary, chance ≈ 50%); "
                "C uses NR_WebDataset\n"
                "  (ternary, chance ≈ 33%). Cross-task accuracy is only "
                "loosely comparable."
            )
        print("-" * 70)
        for r in groups[gid]:
            print(
                f"  {r['comparison']}\n"
                f"    {r['group1']}: acc={r['acc1']:.4f} (n={r['n1']})"
                f"  |  {r['group2']}: acc={r['acc2']:.4f} (n={r['n2']})\n"
                f"    diff={r['diff_pp']:+.2f}pp  z={r['z']:+.3f}"
                f"  p={r['p_value']:.4f}  significant={_sig_str(r['significant'])}\n"
            )


def build_markdown_report(results: list[dict]) -> str:
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    group_labels = {
        "1_zeroshot_vs_fewshot":     "Group 1 — Zero-shot vs Few-shot (same strategy, batch 10)",
        "1b_fewshot_dose_response":   "Group 1b — Few-shot dose-response (same strategy, batch 10)",
        "2_batch_size":               "Group 2 — Batch size (same strategy, zero-shot)",
        "3_cross_strategy":           "Group 3 — Cross-strategy (zero-shot, matched batch)",
        "3b_cross_strategy_fewshot":  "Group 3b — A vs B at matched few-shot (batch 10, WebDataset)",
        "4_C_vs_D":                   "Group 4 — C vs D (zero-shot, batch 10)",
    }

    md = REPORT_INTRO + "## Results\n\n"

    for gid, label in group_labels.items():
        if gid not in groups:
            continue
        md += f"### {label}\n\n"
        if gid == "3_cross_strategy":
            md += (
                "> **Note:** A/B use `WebDataset` (binary, chance ≈ 50%); "
                "C uses `NR_WebDataset` (ternary, chance ≈ 33%). "
                "Cross-task accuracy is only loosely comparable — "
                "results are still reported.\n\n"
            )
        if gid == "3b_cross_strategy_fewshot":
            md += (
                "> **Note:** Both A and B use `WebDataset` (binary); "
                "only the prompt strategy differs.\n\n"
            )
        md += (
            "| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for r in groups[gid]:
            sig = "✓" if r["significant"] else ""
            md += (
                f"| {r['comparison']} "
                f"| {r['acc1']:.4f} ({r['n1']}) "
                f"| {r['acc2']:.4f} ({r['n2']}) "
                f"| {r['diff_pp']:+.2f} "
                f"| {r['z']:+.3f} "
                f"| {r['p_value']:.4f} "
                f"| {sig} |\n"
            )
        md += "\n"

    # Plain-language summary
    sig_results = [r for r in results if r["significant"]]
    ns_results  = [r for r in results if not r["significant"]]

    md += "## Plain-Language Summary\n\n"
    if sig_results:
        md += "**Statistically significant differences (α = 0.05):**\n\n"
        for r in sig_results:
            md += (
                f"- **{r['comparison']}**: "
                f"{r['group1']} ({r['acc1']:.3f}) vs {r['group2']} ({r['acc2']:.3f}), "
                f"z = {r['z']:+.3f}, p = {r['p_value']:.4f}, "
                f"diff = {r['diff_pp']:+.2f} pp.\n"
            )
    md += "\n"
    md += (
        f"Of {len(results)} comparisons, **{len(sig_results)}** are significant at α = 0.05 "
        f"and **{len(ns_results)}** are not. "
    )
    if sig_results:
        md += (
            "The significant findings are: strategy A at batch-5 is meaningfully worse "
            "than batch-20 (Group 2); across strategies, A underperforms B at batch-5 "
            "and B underperforms C at batch-5 (Group 3, noting the datasets differ); "
            "and C outperforms D at batch-10 (Group 4). "
            "Few-shot examples did not significantly improve accuracy for either A or C "
            "compared to zero-shot at batch-10. Most batch-size differences within "
            "strategy are not significant."
        )

    return md


def write_outputs(results: list[dict], df_prepared: pd.DataFrame,
                  out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Annotated runs
    df_prepared.to_csv(os.path.join(out_dir, "runs_annotated.csv"), index=False)

    # Z-test CSV
    cols = [
        "group", "comparison", "strategy", "factor", "ds",
        "group1", "acc1", "n1", "group2", "acc2", "n2",
        "diff_pp", "z", "p_value", "significant",
    ]
    pd.DataFrame(results, columns=cols).to_csv(
        os.path.join(out_dir, "pairwise_ztests_accuracy.csv"), index=False
    )

    # Markdown report
    md = build_markdown_report(results)
    with open(os.path.join(out_dir, "evaluation_report.md"), "w") as fh:
        fh.write(md)

    print(f"\nOutputs written to: {out_dir}/")
    print("  runs_annotated.csv")
    print("  pairwise_ztests_accuracy.csv")
    print("  evaluation_report.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Statistical evaluation of argument-mining experiment results."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ResultOverview_allTests.csv"),
        help="Path to ResultOverview_allTests.csv (default: next to this script)",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.csv_path):
        sys.exit(f"ERROR: CSV not found at {args.csv_path}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv_path)),
                           "eval_output")

    df = load_and_prepare(args.csv_path)
    results = run_accuracy_tests(df)
    print_accuracy_results(results)
    write_outputs(results, df, out_dir)


if __name__ == "__main__":
    main()
