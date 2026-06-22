# Findings: gemma4 on Relation-based Argument Mining, compared to Gorur, Rago and Toni (COLING 2025)

This document summarises what we found in our experiments with gemma4 on the Web-content dataset, what the reference paper reports, and where the two agree or differ. All paper values are micro-F1 in percent, taken from the Web row of the relevant table. Micro-F1 equals accuracy for single-label classification, so our accuracy lines up directly with the paper's "both" (binary) and "all" (ternary) columns. Macro-F1 is the unweighted mean of the per-class F1 scores and is used only descriptively, because it is not a proportion and cannot be significance-tested without the paper's item-level predictions.

We compare against the paper's best model per task on the Web row, as agreed. For binary that is Mixtral-8x7B, for ternary that is Llama2-70B.

## 1. Setup in one paragraph

We tested the open-source model gemma4 (locally hosted) on the same Web-content corpus the paper uses (1348 support, 1316 attack, 1394 neither, 4058 total, paper Table 1). Strategy B is the binary task (attack vs support) and matches the paper's binary RbAM. Strategy C is the ternary task (attack, support, neither) and matches the paper's ternary RbAM. Each run is a fresh random sample of about 1000 pairs, and each configuration was run once. Significance for accuracy is tested with a two-proportion z-test at alpha 0.05.

## 2. Main findings of the paper

The paper makes five core claims for the Web dataset and across its eleven datasets.

First, the strongest LLMs beat the fine-tuned RoBERTa baselines on both tasks. Mixtral-8x7B is the best model overall, and Mistral-7B and Llama2-70B also surpass the baselines on binary. On the Web row, the best binary LLMs reach micro-F1 69, and the best RoBERTa baseline reaches 67.

Second, the binary task is much easier than the ternary task. Averaged over the LLMs, the Web micro-F1 falls from 67 on binary (Table 3) to 51 on ternary (Table 4), a drop of about 16 points.

Third, more informative examples in the primer give better results for the open-source models. The paper concludes that few-shot priming helps, with the exception of GPT-3.5-turbo, which did best zero-shot.

Fourth, the approach is robust to the choice of primer, which the ablation studies in Tables 5 and 6 support.

Fifth, the ternary task is hard for every model. The best ternary result on the Web row is Llama2-70B at micro-F1 54 with the primer (Table 4) and 58 zero-shot (Table 6), and the per-class scores show that the neither class is the easiest of the three for all models.

## 3. Our findings with gemma4

We report only the statistically significant results and the benchmark comparisons that follow from them.

### 3.1 Binary parity reached zero-shot

gemma4 on the binary task reaches micro-F1 about 0.70 and macro-F1 0.70 (best run B 20-shot, batch 10). The zero-shot run already reaches micro-F1 0.696 and macro-F1 0.700 (B 0-shot, batch 5). This matches the paper's best primed binary result (Mixtral-8x7B, micro-F1 69, macro-F1 69.5) and is above the paper's best zero-shot binary result (Mixtral-8x7B, micro-F1 67, Table 5). The interesting part is that gemma4 needs no examples to get there, while the paper's best models needed the primer.

### 3.2 Strong outperformance on ternary

gemma4 on the ternary task reaches micro-F1 0.641 and macro-F1 0.638 (best run C 20-shot, batch 10), with the zero-shot run close behind at micro-F1 0.640 and macro-F1 0.633. The paper's best ternary model on Web is Llama2-70B at micro-F1 54 (primed) and 58 (zero-shot). gemma4 is about 6 to 10 points higher, and it beats the best RoBERTa baseline on Web (RoB-UKP, micro-F1 40) by a wide margin.

### 3.3 The expressiveness trade-off is real and significant

Moving from the binary task to the ternary task lowers accuracy. At the matched configuration (zero-shot, batch 5) binary scores 0.696 and ternary scores 0.624, and the difference is significant (z 3.37, p 0.0007). This confirms that adding the neither class makes the task harder for gemma4 as well.

### 3.4 Adding a relevance instruction costs accuracy

When we extend the ternary prompt with a five-level relevance score (Strategy D), accuracy drops from 0.640 to 0.594 at the same configuration (zero-shot, batch 10), and the difference is significant (z 2.10, p 0.035). Instructing the model to also score relevance measurably degrades the relation labels. This is an important baseline for the planned QBAF stage, because the relevance signal does not come for free.

### 3.5 Few-shot priming gives no significant gain

No few-shot configuration beats its zero-shot counterpart at a significance level of 0.05. The closest cases are A 10-shot vs 0-shot (p 0.071) and B 10-shot vs 20-shot (p 0.055), and the ternary few-shot runs are essentially flat (all p above 0.67). For gemma4 on this corpus, adding examples does not help.

### 3.6 Batch size has no robust effect

Only one batch comparison is significant, A from batch 5 to batch 20 (p 0.046), and it is not reproduced in B (p 0.96) or C. We therefore read this as sampling noise rather than a real mechanism. Practically, packing more pairs per prompt to save calls does not degrade accuracy.

### 3.7 More balanced per-class behaviour on ternary

gemma4's ternary attack-F1 is about 0.58 to 0.61, higher than every model in the paper on Web (attack-F1 there ranges from 0.37 for Mixtral to 0.54 for Llama2). The neither class is the easiest for gemma4 too, with per-class F1 around 0.70.

## 4. Findings we have in common with the paper

We agree on the direction of the expressiveness trade-off. Both the paper and gemma4 show that the binary task is easier than the ternary task. The paper drops from 67 to 51 on average, and gemma4 drops from about 70 to about 64.

We agree that strong open-source LLMs beat the fine-tuned RoBERTa baselines. The paper's best models beat the baselines on both tasks, and gemma4 does the same, reaching about 70 on binary against the 67 baseline and about 64 on ternary against the 40 baseline.

We agree on the per-class ranking within the ternary task. In both the paper and our results, the neither class has the highest per-class F1, so neither is the easiest of the three classes, not the hardest. This is a confirmation of the paper, not a contrast, and we should present it as such.

## 5. Findings where we differ from the paper

We differ on the value of few-shot priming. The paper concludes that more examples help the open-source models, while gemma4 shows no significant benefit from few-shot priming in any strategy. gemma4 also reaches its best binary level zero-shot, whereas the paper's best binary models needed the primer.

We differ on the size of the binary-to-ternary drop. The paper's best models lose about 15 to 21 points when moving from binary to ternary on Web (Mixtral 69 to 48, Llama2 69 to 54). gemma4 loses only about 6 points (0.70 to 0.64). gemma4 degrades far less when the task becomes more expressive, which suggests it handles the harder attack and neither distinctions more gracefully.

We differ on absolute ternary performance. gemma4 is clearly stronger on the ternary Web task than any LLM the paper reports, and its per-class profile is more balanced, in particular on the attack class.

## 6. A clean story to tell

The narrative that the numbers support is the following. A newer, smaller open-source model reaches the level of the paper's best models on the easy binary task without needing any examples, and it clearly exceeds them on the harder ternary task while losing far less accuracy when the task becomes more expressive. The expressiveness trade-off the paper identified still holds, but it is much gentler for gemma4. Two design lessons follow for our pipeline. Few-shot priming and larger batch sizes are not worth their cost here, and adding a relevance instruction to the relation prompt has a measurable accuracy cost that we must account for when we build the QBAF stage.

## 7. Limitations

Each configuration was run once, so our tests reflect the sampling noise of a single draw of about 1000 pairs, not the run-to-run variance of the model. Repeated runs with different seeds would let us separate the two and use variance-based tests.

The effective sample size is below 1000 for some runs because failed batch calls drop their pairs. The ternary zero-shot batch-20 run evaluated 799 pairs and the relevance run (D) evaluated 988. We use the effective n in all tests, but the smaller n widens the confidence intervals.

The comparison with the paper is a cross-model comparison on different samples. We compare gemma4 against Mixtral-8x7B and Llama2-70B, and our runs are random subsets while the paper evaluates the full Web set. The z-test answers whether two proportions differ, not whether everything except the model was identical.

The few-shot comparison is not a fully like-for-like comparison. Both sides are class-balanced. Our sampling splits the examples roughly equally across the eligible classes (attack and support for B, attack, support and neither for C), which matches the balanced design of the paper's primer (2A2S for binary, 1A1S1N for ternary). The differences are elsewhere. First, the count differs. The paper uses a small primer of 3 to 4 examples, while our few-shot runs use 7 to 20 examples, so no run matches the paper's example count exactly. Second, the paper fixed its primer examples across runs, while our examples are resampled on every run. The cleanest comparison is therefore still the zero-shot setting, which is where our strongest binary claim sits.

Macro-F1 is compared descriptively only. It is not a proportion, so it has no simple significance test, and we cannot bootstrap the paper's macro-F1 because we do not have its item-level predictions. The paper also never prints macro-F1, so we compute it from the per-class F1 in Tables 3 and 4, which exist only for the primed configurations.

Accuracy across the binary and ternary tasks is only loosely comparable because the chance level differs (about 0.50 for two classes and about 0.33 for three). For cross-task statements we rely on per-class F1 and macro-F1 rather than raw accuracy.

## Appendix: source map for the key numbers

| Value | Source |
|---|---|
| Web dataset counts 1348 / 1316 / 1394 / 4058 | Paper Table 1, p. 8521 |
| Paper binary best LLM (Mixtral-8x7B) 69 / 70 / 69 | Paper Table 3, p. 8523, Web row, Mixtral-8x7B column |
| Paper binary best baseline (RoB-Kialo) 67 | Paper Table 2, p. 8523, Web row, RoBKialo column |
| Paper binary best zero-shot (Mixtral) 67 | Paper Table 5, p. 8524, Web row, Mixtral column "0" |
| Paper ternary best LLM (Llama2-70B) 53 / 54 / 55 / 54 | Paper Table 4, p. 8524, Web row, Llama2-70B column |
| Paper ternary best baseline (RoB-UKP) 40 | Paper Table 4, p. 8524, Web row, RoBUKP column |
| Paper ternary best zero-shot (Llama2) 58 | Paper Table 6, p. 8524, Web row, Llama2-70B column "0" |
| gemma4 binary 0.696 / 0.700 (0-shot, b5) | ResultOverview run 6/8/26 9:52; accuracy as acc1 in the pairwise file |
| gemma4 binary 0.700 / 0.700 (20-shot, b10) | ResultOverview run 6/8/26 13:50 |
| gemma4 ternary 0.640 / 0.633 (0-shot, b10) | ResultOverview run 6/9/26 23:16 |
| gemma4 ternary 0.641 / 0.638 (20-shot, b10) | ResultOverview run 6/7/26 21:19 |
| B vs C significant (z 3.37, p 0.0007) | pairwise file, group 3_cross_strategy, "B vs C (b5, 0-shot)" |
| C vs D significant (z 2.10, p 0.035) | pairwise file, group 4_C_vs_D |
| Few-shot not significant (best p 0.055 to 0.071) | pairwise file, groups 1 and 1b |
| Batch effect significant only in A (p 0.046) | pairwise file, group 2_batch_size |
