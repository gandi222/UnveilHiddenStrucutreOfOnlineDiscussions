# Findings: Gemma4 Argument Mining Experiments (Macro F1)

Model: `gemma4`. Dataset variants: `WebDataset` (binary, no-relation pairs removed) and `NR_WebDataset` (ternary, no-relation pairs kept). Sample size n=1000 per run, drawn independently from the underlying pool. Reference paper: Gorur, Rago, Toni (2025), *Can Large Language Models perform Relation-based Argument Mining?* (Web-content dataset, full set n≈4058).

All metrics reported are **macro F1** (mean of per-class F1). Where the raw CSV macro F1 column was wrong because empty-prediction rows got bucketed as `no_relation` (rows 8 and 10), the corrected values from the CSV `comment` field are used.

## 1. Strategy definitions (from `prompts.py`)

A — binary, predicts attack only (`{attack: 0/1}`). WebDataset.
B — two-class, predicts attack OR support (`{attack, support}`). WebDataset.
C — three-class, predicts attack / support / neither. NR_WebDataset.
D — three-class + relevance score (QBAF priming). NR_WebDataset.

Comparability rule: only A↔B and C↔D are directly comparable, since the underlying dataset filter (no-relation present/absent) differs.

## 2. Headline results

### Binary (WebDataset)

| Run | Strategy | Few-shot | Chunk | Macro F1 |
|-----|----------|----------|-------|----------|
| 1   | A | 0  | 5  | 0.640 |
| 2   | B | 0  | 5  | **0.700** |
| 3   | A | 0  | 20 | 0.683 |
| 4   | B | 0  | 20 | 0.697 |
| 5   | A | 7  | 10 | 0.669 |
| 6   | B | 10 | 10 | 0.660 |
| 7   | A | 14 | 10 | 0.669 |
| 8   | B | 20 | 10 | **0.700** |
| 9   | A | 10 | 10 | 0.687 |
| 10  | A | 20 | 10 | 0.672 |
| 11  | A | 0  | 10 | 0.649 |

### Ternary (NR_WebDataset)

| Run | Strategy | Few-shot | Chunk | Macro F1 |
|-----|----------|----------|-------|----------|
| 1   | C | 0  | 5  | 0.618 |
| 2   | C | 0  | 20 | 0.606 |
| 3   | C | 10 | 10 | 0.626 |
| 4   | C | 20 | 10 | **0.638** |
| 5   | D | 0  | 10 | 0.599 |
| 6   | C | 0  | 10 | 0.633 |

## 3. Statistical significance (two-proportion z-tests on macro F1, n=1000 each)

Significance codes: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant.

### Within-condition comparisons

| Comparison | Δ F1 | z | p | sig |
|---|---|---|---|---|
| NR · C 0-shot: chunk 5 vs 20 | +0.012 | 0.57 | 0.57 | ns |
| NR · C chunk 10: 0-shot vs 10-shot | +0.007 | 0.30 | 0.76 | ns |
| NR · C chunk 10: 0-shot vs 20-shot | −0.006 | −0.26 | 0.79 | ns |
| NR · C chunk 10: 10-shot vs 20-shot | −0.012 | −0.57 | 0.57 | ns |
| NR · C vs D (chunk 10, 0-shot, QBAF) | +0.033 | 1.53 | 0.13 | ns |
| **Web · A vs B (0-shot, chunk 5)** | **−0.060** | **−2.85** | **0.004** | **\*\*** |
| Web · A vs B (0-shot, chunk 20) | −0.014 | −0.67 | 0.50 | ns |
| **Web · A 0-shot: chunk 5 vs 20** | **−0.043** | **−2.03** | **0.042** | **\*** |
| Web · A chunk 10: 0-shot vs 10-shot | −0.038 | −1.81 | 0.071 | ns (trend) |
| Web · A chunk 10: 0-shot vs 20-shot | −0.023 | −1.09 | 0.28 | ns |
| Web · A chunk 10: 10-shot vs 20-shot | +0.015 | 0.72 | 0.47 | ns |
| Web · B chunk 5 vs 20 (0-shot) | +0.003 | 0.15 | 0.88 | ns |
| Web · B chunk 10: 10-shot vs 20-shot | −0.040 | −1.92 | 0.054 | ns (trend) |

### Versus the Gorur et al. (2025) Web-content baselines (macro F1)

Paper macro F1 computed from the per-class F1 in Tables 3 and 4 (binary: support/attack averaged; ternary: support/attack/neither averaged).

**Binary Web**

| Method | Paper macro F1 | Gemma4 best (B, 20-shot, chunk 10) | z | p | sig |
|---|---|---|---|---|---|
| Llama2-70B | 0.690 | 0.700 | +0.48 | 0.63 | ns |
| Mixtral-8x7B | 0.695 | 0.700 | +0.24 | 0.81 | ns |
| Mistral-7B | 0.650 | 0.700 | +2.39 | 0.017 | \* |
| GPT-3.5-turbo | 0.625 | 0.700 | +3.55 | 0.0004 | \*\*\* |
| RoBERTa (Kialo-tuned, best baseline) | 0.670 | 0.700 | +1.44 | 0.15 | ns |

**Ternary Web**

| Method | Paper macro F1 | Gemma4 best (C, 20-shot, chunk 10) | z | p | sig |
|---|---|---|---|---|---|
| Llama2-70B | 0.540 | 0.638 | +4.49 | <0.001 | \*\*\* |
| Mistral-7B | 0.480 | 0.638 | +7.22 | <0.001 | \*\*\* |
| Mixtral-8x7B | 0.417 | 0.638 | +10.18 | <0.001 | \*\*\* |
| GPT-3.5-turbo | 0.520 | 0.638 | +5.39 | <0.001 | \*\*\* |
| RoBERTa (UKP-tuned, best baseline) | 0.373 | 0.638 | +12.28 | <0.001 | \*\*\* |

### Multiple gemma4 runs vs paper (binary Web)

| gemma4 run | F1 | vs Llama2-70B 0.690 | vs Mixtral-8x7B 0.695 | vs Mistral-7B 0.650 | vs GPT-3.5-turbo 0.625 | vs RoB-Kialo 0.670 |
|---|---|---|---|---|---|---|
| **B, 20s, ch10 (best)** | 0.700 | +0.010, ns | +0.005, ns | +0.050, \* | +0.075, \*\*\* | +0.030, ns |
| B, 0s, ch5 | 0.700 | +0.010, ns | +0.005, ns | +0.050, \* | +0.075, \*\*\* | +0.030, ns |
| B, 0s, ch20 | 0.697 | +0.007, ns | +0.002, ns | +0.047, \* | +0.072, \*\*\* | +0.027, ns |
| A, 10s, ch10 (best A) | 0.687 | −0.003, ns | −0.008, ns | +0.037, ns | +0.062, \*\* | +0.017, ns |
| A, 0s, ch20 | 0.683 | −0.007, ns | −0.012, ns | +0.033, ns | +0.058, \*\* | +0.013, ns |
| A, 0s, ch5 (worst) | 0.640 | −0.050, \* | −0.055, \*\* | −0.010, ns | +0.015, ns | −0.030, ns |

### Multiple gemma4 runs vs paper (ternary Web)

| gemma4 run | F1 | vs Llama2-70B 0.540 | vs Mistral-7B 0.480 | vs Mixtral-8x7B 0.417 | vs GPT-3.5-turbo 0.520 | vs RoB-UKP 0.373 |
|---|---|---|---|---|---|---|
| **C, 20s, ch10 (best)** | 0.638 | +0.098, \*\*\* | +0.158, \*\*\* | +0.222, \*\*\* | +0.118, \*\*\* | +0.265, \*\*\* |
| C, 0s, ch10 | 0.633 | +0.093, \*\*\* | +0.153, \*\*\* | +0.216, \*\*\* | +0.113, \*\*\* | +0.259, \*\*\* |
| C, 10s, ch10 | 0.626 | +0.086, \*\*\* | +0.146, \*\*\* | +0.209, \*\*\* | +0.106, \*\*\* | +0.253, \*\*\* |
| C, 0s, ch5 | 0.618 | +0.078, \*\*\* | +0.138, \*\*\* | +0.201, \*\*\* | +0.098, \*\*\* | +0.245, \*\*\* |
| C, 0s, ch20 (worst C) | 0.606 | +0.066, \*\* | +0.126, \*\*\* | +0.189, \*\*\* | +0.086, \*\*\* | +0.232, \*\*\* |
| D, 0s, ch10 (QBAF) | 0.599 | +0.059, \*\* | +0.119, \*\*\* | +0.183, \*\*\* | +0.079, \*\*\* | +0.226, \*\*\* |

## 4. Findings

1. **The two-class prompt (B) beats attack-only (A) when context is short.** At chunk 5 the gap is 0.060 macro F1 in B's favour (z=−2.85, p=0.004). At chunk 20 the gap shrinks to 0.014 (ns), suggesting Strategy A catches up once it sees more pairs per call. Forcing the model to state both attack and support explicitly removes the ambiguity that surfaces when only the attack field is asked for.

2. **Larger chunks help Strategy A; B is already saturated.** For A, chunk 5 vs chunk 20 at 0-shot is significant (0.640 vs 0.683, p=0.042). For B the same comparison is flat (0.700 vs 0.697, ns). Once the prompt is unambiguous about both labels, batch size stops mattering.

3. **QBAF priming (D) trends below ternary classification (C) but does not reach significance on F1.** D scores 0.033 macro F1 below C at the matched setting (p=0.13). The extra 5-level relevance rubric appears to compete with the relation decision rather than reinforce it. Note: on accuracy the same contrast was significant (p=0.034) because D over-predicts the easy *neither* class, which hits accuracy harder than per-class F1.

4. **Few-shot priming is essentially flat in this regime.** None of the 0-shot vs 10-shot vs 20-shot contrasts (held at chunk 10) reaches p<0.05 on F1 for either C, A or B. The two strongest trends, A 0→10-shot (+0.038, p=0.071) and B 10→20-shot (+0.040, p=0.054), are suggestive but not significant at n=1000. The proposal's expectation that primer size would dominate is not supported here.

5. **Gemma4 matches the paper's best LLMs on binary Web.** The best binary macro F1 (0.700) is statistically indistinguishable from Llama2-70B (0.690) and Mixtral-8x7B (0.695) and from the best RoBERTa baseline (0.670). It is significantly above Mistral-7B (0.650, p=0.017) and GPT-3.5-turbo (0.625, p<0.001). Notable because gemma4 is small and locally hosted, while the paper's strongest models are 8x7B and 70B parameters. The thesis claim should be "matches state-of-the-art LLMs and baseline", not "beats".

6. **Gemma4 substantially outperforms the paper on ternary Web.** The best ternary macro F1 (0.638) is +0.10 to +0.27 above every LLM and baseline in the paper, all at p<0.001. Even Strategy D, the weakest gemma4 ternary setting, is significantly above every paper model (worst case p=0.007 vs Llama2-70B). Two caveats: (a) the sample is n=1000 vs the paper's n≈4058, and (b) the user-side dataset is a re-export of Web-content via Hugging Face, so the exact pair distribution may differ. Even with both caveats, the gap is large enough to indicate a real improvement and is the most thesis-relevant finding.

7. **The "neither" class is the easiest, not the hardest.** In the best ternary run (C, 20-shot, chunk 10) F1 by class is attack 0.61, support 0.61, neither 0.70. The paper's Llama2-70B on Web ternary is essentially balanced across classes (0.53/0.54/0.55). The gain on ternary therefore comes from much better no-relation detection, not from better attack/support discrimination. This is consistent with the binary numbers being on par with the paper.

## 5. Caveats

- All comparisons use macro F1 with a normal approximation to the F1 difference (SE = sqrt(F1·(1−F1)/n)). For F1 values in the 0.50–0.70 range with n=1000 this approximation is adequate but not exact.
- Each test was run once at n=1000. Variance estimates are based on the binomial assumption; no run-to-run variance is captured.
- The paper's Web numbers are on the full dataset (n≈4058); the gemma4 numbers sample 1000 pairs independently per run from a re-exported version of the same dataset. Variance of the gemma4 estimates is roughly 2× the paper's.
