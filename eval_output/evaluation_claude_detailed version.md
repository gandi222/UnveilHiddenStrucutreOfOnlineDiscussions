# Evaluation of `ResultOverview_allTests.csv`

**Model under test:** `gemma4:latest` (open-source, locally hosted via Ollama)
**Dataset:** Web-content corpus (`WebDataset` = binary subset, 2 664 pairs; `NR_WebDataset` = full set, 4 058 pairs incl. *neither*)
**Reference paper:** Gorur, Rago & Toni, *Can Large Language Models perform Relation-based Argument Mining?*, COLING 2025.
**Per run:** `LIMIT = 1000`, so each row is a fresh random sample of 1 000 pairs (≈ a proportion measured on n = 1 000).

---

## 1. What the labels mean and which comparisons are valid

From the project README and `prompts.py`, the four strategies differ only in how many classes the model is asked to produce:

| Strategy | Task | Classes | Paper equivalent |
|---|---|---|---|
| **A** | "Does Arg2 attack Arg1?" | attack / not-attack (binary, 1 field) | **none** (paper never frames RbAM as attack-detection) |
| **B** | attack **vs** support | 2-class | **Binary RbAM** |
| **C** | attack / support / neither | 3-class | **Ternary RbAM** |
| **D** | C **+ 5-level relevance score** | 3-class + score | C (the relevance score is the QBAF extension; not scored here) |

Dataset distinction: **A and B run on `WebDataset`** (the binary subset — only attack+support, no *neither*). **C and D run on `NR_WebDataset`** (the same web-content corpus, but with the 1 394 *neither* pairs added back). They are the *same underlying data*; NR just re-includes the no-relation class. This matters because chance level differs (≈50 % for a 2-class task, ≈33 % for a 3-class task), so raw accuracy is **not** directly comparable across A/B and C — only *within* a task, or descriptively across tasks.

**Comparisons that are meaningful** (vary exactly one factor, hold task + dataset fixed):

1. Few-shot vs zero-shot, same strategy, same batch size.
2. Batch size, same strategy, zero-shot.
3. A vs B vs C — expressiveness trade-off (with the dataset caveat above).
4. C vs D — cost of adding the relevance instruction, same config.
5. Each of A/B/C against the paper's matching table.

**A data-quality fix first.** For strategy B the CSV's `macro_f1` column (0.466, 0.464) is wrong: it averages in a spurious *no_relation* F1 = 0, even though B cannot predict that class and `WebDataset` contains no *neither* pairs. Your own comments in the CSV flag this. The correct 2-class macro-F1 (mean of attack-F1 and support-F1) is **0.700 / 0.697** for the two zero-shot runs — which matches the values you noted by hand (0.6998, 0.696663). All B figures below use the corrected 2-class macro-F1. For A, the reported `macro_f1` is already a correct 2-class (attack / not-attack) macro.

---

## 2. Statistical approach — and which test is the right one

Each run reports one accuracy. **Accuracy is a proportion** (correct ÷ evaluated), and that single fact drives the whole choice of test.

**Effective sample size (important correction).** The denominator is *not* always 1 000. When a batch API call fails, every pair in it is dropped, so the real n is the number of non-NA predictions (the sum of the `predicted_count_*` columns). Two runs are materially affected — **C zero-shot batch-20 actually evaluated 799 pairs** (not 1 000) and **D evaluated 988** — plus a few others off by 1–100. All tests below use the **effective n**, not the nominal 1 000.

**Why a z-test (and what it is).** A two-proportion z-test asks whether two proportions differ. Each accuracy `p = x/n` is a binomial proportion; for large n the sampling distribution of `p` is approximately normal (Central Limit Theorem), so the difference `p₁ − p₂` divided by its standard error follows a standard normal ("z") distribution. With n ≈ 800–1 000 the approximation is excellent. It is equivalent to a χ² test on the 2×2 (correct/incorrect) × (config A/config B) table — `z² = χ²` exactly, which the verification confirms — so "z-test" and "chi-square test of two proportions" are the same result reported two ways.

**Can you use a t-test? — No, not for these single-run accuracies.** A t-test compares the *means of continuous measurements* and uses the sample standard deviation (estimated from replicates) with a t-distribution to absorb the uncertainty of that estimate. Here the outcome is binary (each pair is right/wrong), and for Bernoulli data the variance is fixed by the mean itself (`p(1−p)`) — there is nothing extra to estimate, so the z-test is the correct large-sample tool and a t-test on a proportion is a category error. A t-test **becomes the right test only if you change the design**: run each configuration several times with different random seeds, collect a *sample* of accuracies per configuration, and then compare those samples with a two-sample (or paired) t-test — e.g. the 5×2-cv t-test recommended by Dietterich (1998). With the current one-run-per-config data there is no such sample, so a t-test is not applicable.

**Can you use a more familiar test than Cochran–Armitage for the trend questions? — Yes.** Cochran–Armitage is simply the standard test for a *trend in a proportion across ordered levels* (batch 5/10/20; few-shot 0/7/10/14/20). It is exactly equivalent to a **logistic regression** of correct/incorrect on the ordered factor and testing the slope — which is the more common, more interpretable formulation (the slope is a log-odds-per-step). The verification ran both and they give identical p-values, so the report keeps the logistic-regression framing. The simplest familiar option is also fine: just do the **pairwise z-tests between adjacent levels** (reported below) — the trend test only adds a single global "is there a monotonic trend" answer on top of them.


Significance threshold α = 0.05, two-sided. F1 differences are compared descriptively (no item-level data to bootstrap). Since each configuration was run **once**, these tests capture the sampling noise of a single draw, not run-to-run model variance — see Caveats.

---

## 3. Results

### 3.1 Zero-shot vs few-shot

**Strategy A (Web, batch 10):** 0-shot 0.649 → 7-shot 0.669 → **10-shot 0.687** → 14-shot 0.669 → 20-shot 0.672.
The best point is 10-shot, but no individual few-shot run beats zero-shot significantly (10-shot vs 0-shot: z = 1.80, p = 0.071) and there is no monotonic trend (Cochran–Armitage z = 1.03, p = 0.30). Few-shot gives at most a small, non-significant lift and saturates/declines past ~10 examples.

**Strategy C (NR_Web, batch 10):** 0-shot 0.640, 10-shot 0.632, 20-shot 0.641 — essentially flat. No pairwise difference (all p > 0.7) and no trend (z = 0.05, p = 0.96). Few-shot examples do **not** help the ternary task here.

**Strategy B (Web):** no zero-shot/few-shot pair exists at a common batch size, so it cannot be tested cleanly. Descriptively, 20-shot batch-10 (0.700) ≈ zero-shot batch-5/20 (0.696/0.697), i.e. again no benefit from priming.

> **Finding:** For gemma4 on this corpus, few-shot priming yields **no statistically significant gain** over zero-shot in any strategy. This *contrasts* with the paper, whose ablations (Tables 5 & 6) conclude "more (informative) examples give better results" for the open-source models — though the paper also found GPT-3.5 did best zero-shot.

### 3.2 Batch size

**Strategy A (Web, zero-shot):** 0.640 (b5) → 0.649 (b10) → 0.683 (b20). Significant rise from b5 to b20 (z = −2.00, p = 0.046) and a significant increasing trend (logistic slope +0.013/step, p = 0.034).

**Strategy C (NR_Web, zero-shot):** 0.624 (b5) → 0.640 (b10) → 0.609 (b20; note effective n = 799). No significant pairwise difference (all p ≥ 0.17), no trend (logistic slope −0.006/step, p = 0.36).

**Strategy B (Web, zero-shot):** b5 0.696 vs b20 0.697 — identical (z = −0.05, p = 0.96).

> **Finding:** Batch size has **no consistent, robust effect**. The one significant signal (A improving with larger batches) is not reproduced in B or C, so it is most likely sampling noise rather than a real mechanism. Practically: batching to reduce API calls does **not** degrade accuracy, which is a useful methodological result for the thesis.

### 3.3 A vs B vs C — the expressiveness trade-off

*(A & B on `WebDataset` binary subset; C on `NR_WebDataset` full set — same web corpus, C additionally must handle the 1 394 *neither* pairs. Accuracy across tasks is only loosely comparable because chance levels differ.)*

| Strategy | Best accuracy | Macro-F1 (corrected) | Task |
|---|---|---|---|
| A (attack-only) | 0.683 | 0.683 | binary, 2-class, chance ≈ 0.50 |
| B (attack/support) | 0.700 | 0.700 | binary, 2-class, chance ≈ 0.50 |
| C (3-class) | 0.641 | 0.638 | ternary, 3-class, chance ≈ 0.33 |

A vs B at matched config: at batch 5 B beats A significantly (0.696 vs 0.640, z = −2.63, p = 0.008); at batch 20 they are equal (0.697 vs 0.683, p = 0.53). B vs C at batch 5: B 0.696 vs C 0.624, z = 3.37, p = 0.0007.

> **Finding:** Accuracy falls as the framework becomes more expressive — A/B (binary) ≈ 0.68–0.70 → C (ternary) ≈ 0.64. Relative to chance the drop is larger still (binary ~0.20 above chance; ternary ~0.31 above its lower chance baseline — so in *absolute* terms gemma4 actually extracts *more* signal in the ternary setting, but headline accuracy is lower). This directly supports thesis **RQ2** (added expressiveness → lower modelling accuracy) and mirrors the paper, where the LLM average on Web drops from **67 (binary, Table 3)** to **51 (ternary, Table 4)**.

### 3.4 C vs D — cost of the relevance instruction

Same config (NR_Web, zero-shot, batch 10): **C 0.640 (n = 1 000) vs D 0.594 (n = 988)**, z = 2.10, **p = 0.035** (macro-F1 0.633 → 0.599). Adding the 5-level relevance rubric to the *same* 3-class prompt **significantly lowers** relation-classification accuracy by ~4.6 points. This is an important baseline for the QBAF step: the relevance/strength signal does not come for free — instructing the model to also score relevance measurably degrades the relation labels. (Only one D run exists, so treat this as indicative, not definitive.)

---

## 4. Comparison with the paper

All paper values are **micro-F1 (%)**. For a single-label task micro-F1 = accuracy, so our `accuracy` is the right column to line up against the paper's "both"/"all" micro-F1; our per-class `f1_score_*` line up against the paper's per-class support/attack/neither F1. The paper's `Web` row = the same Web-content dataset (Table 1, p. 8520: 1 348 support / 1 316 attack / 1 394 neither / 4 058 total — identical to `NR_WebDataset`).

### 4.1 Binary (our Strategy B vs paper Binary RbAM)

**Compared against:** **Table 3** ("F1 scores … for the models used … with 2A2S", **p. 8523**), row **Web**; baseline **Table 2** (RoBERTa, **p. 8523**), row **Web**; ablation **Table 5** ("ablation studies for the binary RbAM task", **p. 8524**), row **Web**, column **0** (zero-shot).

| Source | support / attack / micro(=acc) |
|---|---|
| **gemma4 — B, 20-shot, b10 (ours)** | 0.703 / 0.697 / **0.700** |
| **gemma4 — B, 0-shot, b5 (ours)** | 0.695 / 0.704 / **0.696** |
| Mixtral-8x7B (Table 3, 2A2S) | 69 / 70 / **69** |
| Llama2-70B (Table 3, 2A2S) | 68 / 70 / **69** |
| Mistral-7B (Table 3, 2A2S) | 60 / 70 / **65** |
| GPT-3.5-turbo (Table 3, 2A2S) | 56 / 69 / **64** |
| RoBERTa, best baseline on Web (Table 2, RoB-Kialo) | 67 / 67 / **67** |
| Best LLM zero-shot on Web (Table 5, Mixtral col "0") | — / — / **67** |

> **gemma4 matches or slightly exceeds the strongest binary LLMs** in the paper on the Web dataset (~70 vs 69 micro-F1) and beats the best RoBERTa baseline (67). Notably gemma4 reaches this **zero-shot** (0.696), whereas the paper's best open models used the 2A2S primer.

### 4.2 Ternary (our Strategy C vs paper Ternary RbAM)

**Compared against:** **Table 4** ("F1 scores … for the baselines … and models … with 1A1S1N", **p. 8524**), row **Web** — LLM columns for the model scores and the left-side RoB columns for the baseline; ablation **Table 6** ("ablation studies for the ternary RbAM task", **p. 8524**), row **Web**.

| Source | support / attack / neither / micro(=acc) |
|---|---|
| **gemma4 — C, 20-shot, b10 (ours)** | 0.607 / 0.608 / 0.700 / **0.641** |
| **gemma4 — C, 0-shot, b10 (ours)** | 0.620 / 0.576 / 0.701 / **0.640** |
| Llama2-70B (Table 4, 1A1S1N) | 53 / 54 / 55 / **54** |
| Mistral-7B (Table 4, 1A1S1N) | 44 / 44 / 56 / **49** |
| Mixtral-8x7B (Table 4, 1A1S1N) | 30 / 37 / 58 / **48** |
| GPT-3.5-turbo (Table 4, 1A1S1N) | 47 / 51 / 58 / **53** |
| Best LLM zero-shot on Web (Table 6, Llama2-70B col "0") | — | **58** |
| RoBERTa, best baseline on Web (Table 4 left, RoB-UKP) | 24 / 14 / 55 / **40** |

> **gemma4 clearly outperforms every LLM in the paper on the ternary Web task** (~64 vs the paper's best 54, or 58 in the zero-shot ablation) and massively beats the best RoBERTa baseline (40). Its per-class profile is also more balanced (attack-F1 ≈ 0.58–0.61 vs the paper's 0.37–0.54), suggesting gemma4 handles the harder attack/neither distinction better than the 2023–24 open models.

### 4.3 Strategy A

No comparison — the paper never frames RbAM as attack-detection (confirmed in `Datasets/datasetStructure.md`). A is kept only as an attack-detection probe; on the binary `WebDataset`, "not-attack" = "support", so A and B partition the same data, which is why their accuracies are similar.

---

## 5. Summary of findings

1. **Few-shot priming gives gemma4 no significant benefit** on any strategy (best case A 10-shot, p = 0.07) — opposite to the paper's "more examples help" conclusion for open models.
2. **Batch size has no robust effect** — accuracy is stable whether 5, 10, or 20 pairs are packed per prompt. Good news for cost-saving batching.
3. **Expressiveness trade-off confirmed (RQ2):** accuracy declines from binary (~0.70) to ternary (~0.64), matching the paper's binary→ternary drop on Web (67→51).
4. **The relevance instruction costs accuracy** — C→D drops 0.640→0.594 (p = 0.034); relevant for designing the QBAF stage.
5. **Versus the paper, gemma4 is competitive on binary and superior on ternary** for the Web dataset, beating both the RoBERTa baselines and the 2023–24 LLMs (Llama2-70B, Mixtral-8x7B, Mistral-7B, GPT-3.5-turbo) — supporting the choice of an open-source model for the replication and the QBAF extension.

### Caveats
- One run per configuration ⇒ tests reflect single-draw sampling noise, not model run-to-run variance. Repeated runs (≥3, varied seeds) per config would let you separate the two and use proper variance-based tests (two-sample / 5×2-cv t-test).
- **Failed batches shrink the effective n** (C zero-shot batch-20 → 799; D → 988). Accuracies are still valid (NA pairs are correctly excluded), but comparisons must use the effective n, as done here. Reducing batch failures (higher `MAX_RETRIES`) would keep all runs at n = 1 000.
- A/B vs C accuracy is only loosely comparable (different class count and chance level); per-class F1 and macro-F1 are the fairer cross-task metrics.
- Paper reports micro-F1; we equate it to accuracy (valid for single-label classification). Our macro-F1 is reported separately and is the stricter, class-balanced view.

### Verification performed
- Recomputed every per-class precision/recall/F1 from the raw TP/FP/FN counts → **0 mismatches** with the CSV.
- Re-derived 3-class accuracy as ΣTP / effective-n → surfaced the failed-batch denominator issue above (the only discrepancy, now corrected).
- Cross-checked all z-tests three ways: hand formula, `statsmodels.proportions_ztest`, and `proportions_chisquare` → identical (and `z² = χ²` confirmed).
- Reproduced every trend test with logistic regression → identical p-values to Cochran–Armitage.
- Confirmed the corrected 2-class macro-F1 for B (0.6998 / 0.6967) matches your hand-noted values in the CSV comments.

### References on test choice
- Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* — recommends McNemar when models can't be retrained/re-run many times.
- Raschka (2018), *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning* (arXiv:1811.12808) — practical guide to McNemar, 5×2-cv t-test, and proportion tests.
