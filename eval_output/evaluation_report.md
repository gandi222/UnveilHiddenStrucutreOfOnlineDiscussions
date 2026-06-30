# Statistical Evaluation Report

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

## Results

### Group 1 — Zero-shot vs Few-shot (same strategy, batch 10)

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| A 7-shot vs 0-shot (b10) | 0.6690 (1000) | 0.6490 (1000) | +2.00 | +0.943 | 0.3455 |  |
| A 10-shot vs 0-shot (b10) | 0.6870 (1000) | 0.6490 (1000) | +3.80 | +1.804 | 0.0712 |  |
| A 14-shot vs 0-shot (b10) | 0.6690 (1000) | 0.6490 (1000) | +2.00 | +0.943 | 0.3455 |  |
| A 20-shot vs 0-shot (b10) | 0.6720 (1000) | 0.6490 (1000) | +2.30 | +1.086 | 0.2774 |  |
| C 10-shot vs 0-shot (b10) | 0.6320 (999) | 0.6400 (1000) | -0.84 | -0.389 | 0.6974 |  |
| C 20-shot vs 0-shot (b10) | 0.6410 (999) | 0.6400 (1000) | +0.06 | +0.030 | 0.9762 |  |

### Group 1b — Few-shot dose-response (same strategy, batch 10)

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| A 7-shot vs 10-shot (b10) | 0.6690 (1000) | 0.6870 (1000) | -1.80 | -0.861 | 0.3890 |  |
| A 7-shot vs 14-shot (b10) | 0.6690 (1000) | 0.6690 (1000) | +0.00 | +0.000 | 1.0000 |  |
| A 7-shot vs 20-shot (b10) | 0.6690 (1000) | 0.6720 (1000) | -0.30 | -0.143 | 0.8865 |  |
| A 10-shot vs 14-shot (b10) | 0.6870 (1000) | 0.6690 (1000) | +1.80 | +0.861 | 0.3890 |  |
| A 10-shot vs 20-shot (b10) | 0.6870 (1000) | 0.6720 (1000) | +1.50 | +0.719 | 0.4723 |  |
| A 14-shot vs 20-shot (b10) | 0.6690 (1000) | 0.6720 (1000) | -0.30 | -0.143 | 0.8865 |  |
| B 10-shot vs 20-shot (b10) | 0.6600 (1000) | 0.7000 (1000) | -4.00 | -1.917 | 0.0552 |  |
| C 10-shot vs 20-shot (b10) | 0.6320 (999) | 0.6410 (999) | -0.90 | -0.419 | 0.6756 |  |

### Group 2 — Batch size (same strategy, zero-shot)

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| A b5 vs b10 (0-shot) | 0.6400 (1000) | 0.6490 (1000) | -0.90 | -0.420 | 0.6742 |  |
| A b5 vs b20 (0-shot) | 0.6400 (1000) | 0.6830 (940) | -4.30 | -1.998 | 0.0457 | ✓ |
| A b10 vs b20 (0-shot) | 0.6490 (1000) | 0.6830 (940) | -3.40 | -1.585 | 0.1129 |  |
| C b5 vs b10 (0-shot) | 0.6240 (1000) | 0.6400 (1000) | -1.60 | -0.742 | 0.4582 |  |
| C b10 vs b20 (0-shot) | 0.6400 (1000) | 0.6088 (799) | +3.17 | +1.382 | 0.1669 |  |
| C b5 vs b20 (0-shot) | 0.6240 (1000) | 0.6088 (799) | +1.57 | +0.682 | 0.4950 |  |
| B b5 vs b20 (0-shot) | 0.6960 (989) | 0.6967 (900) | -0.10 | -0.048 | 0.9618 |  |

### Group 3 — Cross-strategy (zero-shot, matched batch)

> **Note:** A/B use `WebDataset` (binary, chance ≈ 50%); C uses `NR_WebDataset` (ternary, chance ≈ 33%). Cross-task accuracy is only loosely comparable — results are still reported.

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| A vs B (b5, 0-shot) | 0.6400 (1000) | 0.6960 (989) | -5.57 | -2.634 | 0.0084 | ✓ |
| A vs B (b20, 0-shot) | 0.6830 (940) | 0.6967 (900) | -1.37 | -0.634 | 0.5258 |  |
| B vs C (b5, 0-shot) | 0.6960 (989) | 0.6240 (1000) | +7.17 | +3.372 | 0.0007 | ✓ |

### Group 3b — A vs B at matched few-shot (batch 10, WebDataset)

> **Note:** Both A and B use `WebDataset` (binary); only the prompt strategy differs.

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| A vs B 10-shot (b10) | 0.6870 (1000) | 0.6600 (1000) | +2.70 | +1.287 | 0.1979 |  |
| A vs B 20-shot (b10) | 0.6720 (1000) | 0.7000 (1000) | -2.80 | -1.349 | 0.1773 |  |

### Group 4 — C vs D (zero-shot, batch 10)

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| C vs D (b10, 0-shot) | 0.6400 (1000) | 0.5940 (988) | +4.59 | +2.104 | 0.0354 | ✓ |

### Group 5 — Cross-model comparison (matched b20 conditions)

| Comparison | acc1 (n1) | acc2 (n2) | diff (pp) | z | p-value | Sig? |
|---|---|---|---|---|---|---|
| B 0-shot Web b20 | gemma4:latest vs claude-opus-4-8 | 0.6967 (900) | 0.7340 (1000) | -3.73 | -1.802 | 0.0715 |  |
| B 0-shot Web b20 | gemma4:latest vs gpt-5.5 | 0.6967 (900) | 0.7900 (1000) | -9.33 | -4.665 | 0.0000 | ✓ |
| B 0-shot Web b20 | claude-opus-4-8 vs gpt-5.5 | 0.7340 (1000) | 0.7900 (1000) | -5.60 | -2.940 | 0.0033 | ✓ |
| B 20-shot Web b20 | claude-opus-4-8 vs gpt-5.5 | 0.7390 (1000) | 0.7780 (1000) | -3.90 | -2.038 | 0.0416 | ✓ |
| C 0-shot NR_Web b20 | gemma4:latest vs claude-opus-4-8 | 0.6088 (799) | 0.6720 (1000) | -6.37 | -2.805 | 0.0050 | ✓ |
| C 0-shot NR_Web b20 | gemma4:latest vs gpt-5.5 | 0.6088 (799) | 0.7020 (1000) | -9.37 | -4.171 | 0.0000 | ✓ |
| C 0-shot NR_Web b20 | claude-opus-4-8 vs gpt-5.5 | 0.6720 (1000) | 0.7020 (1000) | -3.00 | -1.447 | 0.1480 |  |
| C 20-shot NR_Web b20 | claude-opus-4-8 vs gpt-5.5 | 0.7170 (1000) | 0.7210 (1000) | -0.40 | -0.199 | 0.8423 |  |

## Plain-Language Summary

**Statistically significant differences (α = 0.05):**

- **A b5 vs b20 (0-shot)**: b5 (0.640) vs b20 (0.683), z = -1.998, p = 0.0457, diff = -4.30 pp.
- **A vs B (b5, 0-shot)**: A (0.640) vs B (0.696), z = -2.634, p = 0.0084, diff = -5.57 pp.
- **B vs C (b5, 0-shot)**: B (0.696) vs C (0.624), z = +3.372, p = 0.0007, diff = +7.17 pp.
- **C vs D (b10, 0-shot)**: C (0.640) vs D (0.594), z = +2.104, p = 0.0354, diff = +4.59 pp.
- **B 0-shot Web b20 | gemma4:latest vs gpt-5.5**: gemma4:latest (0.697) vs gpt-5.5 (0.790), z = -4.665, p = 0.0000, diff = -9.33 pp.
- **B 0-shot Web b20 | claude-opus-4-8 vs gpt-5.5**: claude-opus-4-8 (0.734) vs gpt-5.5 (0.790), z = -2.940, p = 0.0033, diff = -5.60 pp.
- **B 20-shot Web b20 | claude-opus-4-8 vs gpt-5.5**: claude-opus-4-8 (0.739) vs gpt-5.5 (0.778), z = -2.038, p = 0.0416, diff = -3.90 pp.
- **C 0-shot NR_Web b20 | gemma4:latest vs claude-opus-4-8**: gemma4:latest (0.609) vs claude-opus-4-8 (0.672), z = -2.805, p = 0.0050, diff = -6.37 pp.
- **C 0-shot NR_Web b20 | gemma4:latest vs gpt-5.5**: gemma4:latest (0.609) vs gpt-5.5 (0.702), z = -4.171, p = 0.0000, diff = -9.37 pp.
- **claude-opus-4-8 C 20-shot vs 0-shot (b20)**: 20-shot (0.717) vs 0-shot (0.672), z = +2.185, p = 0.0289, diff = +4.50 pp.

Of 39 comparisons, **10** are significant at α = 0.05 and **29** are not. The significant findings are: strategy A at batch-5 is meaningfully worse than batch-20 (Group 2); across strategies, A underperforms B at batch-5 and B underperforms C at batch-5 (Group 3, noting the datasets differ); and C outperforms D at batch-10 (Group 4). Few-shot examples did not significantly improve accuracy for either A or C compared to zero-shot at batch-10. Most batch-size differences within strategy are not significant.