# Argument Relation Evaluation with Ollama

LLM-based evaluation of argument pair relationships using a locally-hosted Ollama server. Multiple argument pairs are packed into a single prompt (batching) to reduce API calls. Four prompt strategies — binary, two-class, three-class, and three-class with relevance scoring — are evaluated independently against a ground-truth dataset. Supports zero-shot and few-shot prompting.

---

## Overview

Given a dataset of argument pairs `(arg1, arg2)`, the system asks an LLM to classify the relation between them. Four increasingly granular strategies are available:

| Strategy | Classes predicted | Output fields |
|----------|-------------------|---------------|
| A — Binary | Attack only | `pred_attack` |
| B — Two-class | Attack or Support | `pred_attack`, `pred_support` |
| C — Three-class | Attack, Support, or No Relation | `pred_attack`, `pred_support`, `pred_neither` |
| D — Three-class + Relevance | Attack, Support, or No Relation + a 5-level relevance score | `pred_attack`, `pred_support`, `pred_neither`, `pred_relevance` |

Ground-truth labels in the dataset (`support` column):

| Value | Meaning |
|-------|---------|
| `0` | Attack |
| `1` | Support |
| `2` | No Relation |

---

## Dataset

The dataset lives in `NR_WebDataset/` in HuggingFace Arrow IPC stream format. Each row has:

- `arg1` — first argument (string)
- `arg2` — second argument (string)
- `support` — ground-truth label (int: 0, 1, or 2)
- `type` — relation type metadata
- `relevance` — relevance score (float, required for strategy D few-shot; optional otherwise)

The dataset contains ~4000 argument pairs. Rows are shuffled before processing so any `LIMIT` slice is a random sample. Few-shot examples are sampled from the dataset and excluded from the evaluation set.

---

## Project Structure

```
.
├── main.py                     # entry point — configuration and orchestration
├── pipeline.py                 # dataset loading, batching, and result saving
├── prompts.py                  # prompt builders for strategies A/B/C/D
├── api_client.py               # Ollama API calls and JSON response parsing
├── evaluation.py               # metrics (accuracy, macro F1, confusion matrix)
├── requirements.txt            # Python dependencies
├── results2.csv                # per-pair predictions (generated on run)
├── ResultOverview_allTests.csv # one row per run with aggregate metrics
└── NR_WebDataset/
    ├── data-00000-of-00001.arrow   # argument pair dataset (Arrow IPC format)
    ├── dataset_info.json           # schema metadata
    └── state.json                  # HuggingFace dataset state
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:
```
langchain-ollama
pyarrow
pandas
```

### 2. Access the Ollama server

The default server is configured in `main.py`:

```python
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
```

No API key is required. Replace `BASE_URL` with your own Ollama instance if needed (e.g. `http://localhost:11434`).

### 3. Run

```bash
python main.py
```

---

## Configuration

All tunable variables are at the top of [main.py](main.py):

```python
ARROW_FILE        = "NR_WebDataset/data-00000-of-00001.arrow"  # input dataset path
OUTPUT_CSV        = "results2.csv"          # output file for per-pair predictions
BASE_URL          = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"  # Ollama server
MODEL             = "gemma4:latest"         # model to use
LIMIT             = 20      # max pairs to evaluate; None = all ~4000
BATCH_SIZE        = 10      # pairs per API call
DELAY_SECONDS     = 0       # pause between API calls
MAX_RETRIES       = 1       # retry attempts per failed API call

STRATEGIES_TO_RUN = ["D"]   # "A", "B", "C", "D", or any combination

FEW_SHOT_N        = 5       # labeled examples per prompt; 0 = zero-shot
TRACK_RESULTS     = 1       # if 1, appends a summary row to ResultOverview_allTests.csv
```

---

## Prompt Strategies

Each strategy packs `BATCH_SIZE` argument pairs into one prompt and asks the model to return a JSON array with one result per pair. An optional few-shot block of labeled examples is inserted between the task description and the pairs to classify.

Few-shot examples are sampled from the dataset balanced across eligible classes (strategy B excludes No Relation examples since it cannot predict that class). These rows are removed from the evaluation set before running.

### Strategy A — Binary

Predicts only whether Arg2 attacks Arg1. Output schema: `{"attack": 0}` or `{"attack": 1}`.

### Strategy B — Two-class

Predicts Attack or Support. Output schema: `{"attack": 0, "support": 0}` with exactly one field set to 1.

### Strategy C — Three-class

Predicts Attack, Support, or No Relation. Output schema: `{"attack": 0, "support": 0, "neither": 0}` with exactly one field set to 1.

### Strategy D — Three-class + Relevance Score

Extends strategy C with a 5-level relevance rubric (0.00 / 0.25 / 0.50 / 0.75 / 1.00) that rates how substantively Arg2 engages Arg1. Output schema: `{"attack": 0, "support": 0, "neither": 0, "relevance": 0.00}`. The relevance score is recorded in `pred_relevance` but excluded from accuracy/F1 computation.

If the dataset has no `relevance` column, strategy D falls back to zero-shot (no few-shot examples can be provided for the relevance field).

#### Relevance Score Rubric 

| Score | Label | Definition used in the prompt |
|-------|-------|-------------------------------|
| `0.00` | Irrelevant | Off-topic; no logical bearing on the parent argument. Includes insults, digressions, and purely emotional reactions. |
| `0.25` | Weak | Loosely related to the parent but lacking substance. Vague restatements, unsupported assertions, or anecdotal remarks without reasoning. |
| `0.50` | Moderate | On-topic and provides a reason or counter-reason, but the reasoning is generic, partial, or applicable to many arguments beyond the specific parent. |
| `0.75` | Strong | Directly engages the parent argument with specific reasoning, evidence, or a well-formed counterexample. |
| `1.00` | Decisive | Targets the core of the parent argument with substantive, specific reasoning that materially affects how the parent should be evaluated. |

The model is instructed to pick exactly one of these five values; any other float is treated as a parse error.

---

## Output

### results2.csv

Per-pair predictions. Each argument pair appears once per strategy run.

| Column | Description |
|--------|-------------|
| `orig_idx` | Original row index in the dataset before shuffling |
| `arg1` | First argument text |
| `arg2` | Second argument text |
| `support [true value]` | Ground-truth label (0=Attack, 1=Support, 2=No Relation) |
| `support_label` | Human-readable ground-truth |
| `strategy` | Which strategy produced this row (A, B, C, or D) |
| `pred_attack` | Model prediction: 1 if attack, 0 if not, `<NA>` if batch failed |
| `pred_support` | Model prediction: 1 if support, 0 if not, `<NA>` if batch failed |
| `pred_neither` | Model prediction: 1 if no relation, 0 if not, `<NA>` if batch failed |
| `pred_relevance` | Relevance score (strategy D only), `<NA>` otherwise |

If a batch API call fails after `MAX_RETRIES` attempts, all rows in that batch receive `<NA>` for prediction columns.

### ResultOverview_allTests.csv

One row is appended per strategy run when `TRACK_RESULTS=1`. Each row is computed from the current `results2.csv` immediately after the run completes (via `evaluation.append_to_overview`).

**Run metadata columns**

| Column | Description |
|--------|-------------|
| `timestamp` | Wall-clock time when the row was written (`YYYY-MM-DD HH:MM:SS`) |
| `git_commit` | Short SHA of the HEAD commit at run time |
| `model` | Ollama model string (e.g. `gemma4:latest`) |
| `strategy` | Strategy letter (`A`, `B`, `C`, or `D`) |
| `few_shot_n` | Number of few-shot examples used (0 = zero-shot) |
| `dataset_file` | Path to the Arrow input file |
| `limit_config` | `LIMIT` setting from `main.py` (`None` = full dataset) |
| `batch_size` | Pairs per API call |
| `delay_seconds` | Pause between API calls |
| `max_retries` | Retry attempts per failed batch |

**Dataset structure columns**

| Column | Description |
|--------|-------------|
| `n_evaluated` | Number of argument pairs evaluated in this run |
| `ground_truth_count_attack` | Pairs with true label Attack (0) |
| `ground_truth_count_support` | Pairs with true label Support (1) |
| `ground_truth_count_no_relation` | Pairs with true label No Relation (2) |
| `predicted_count_attack` | Pairs the model predicted as Attack |
| `predicted_count_support` | Pairs the model predicted as Support |
| `predicted_count_no_relation` | Pairs the model predicted as No Relation |

**Metric columns**

All metrics treat each class as the positive class in a one-vs-rest setup:

| Column | Description |
|--------|-------------|
| `accuracy` | `correct / n_evaluated` (rows with `<NA>` predictions excluded from denominator) |
| `true_positive_<class>` | Model predicted `<class>` and ground truth is `<class>` |
| `false_positive_<class>` | Model predicted `<class>` but ground truth is not `<class>` |
| `false_negative_<class>` | Ground truth is `<class>` but model predicted something else |
| `true_negative_<class>` | Ground truth is not `<class>` and model did not predict `<class>` |
| `precision_<class>` | `TP / (TP + FP)`; 0 when denominator is 0 |
| `recall_<class>` | `TP / (TP + FN)`; 0 when denominator is 0 |
| `f1_score_<class>` | `2 · precision · recall / (precision + recall)`; 0 when denominator is 0 |
| `macro_f1` | Unweighted average of per-class F1 scores |

`<class>` is one of `attack`, `support`, or `no_relation`. For strategy A, `no_relation` and `support` metrics are `NaN` (that class is not predicted).

**How ResultOverview rows are calculated**

After each strategy run, `evaluation.append_to_overview()` re-reads `results2.csv`, filters to rows for the current strategy, and computes all metrics from scratch:

1. Each row is mapped to a `(y_true, y_pred)` label pair using the same correctness rules as the console output (strategy A: binary Attack/Not-Attack; B: Attack or Support with No Relation always wrong; C/D: three-class).
2. Rows where any required prediction column is `<NA>` (failed batches) are excluded from F1 and accuracy denominators.
3. Per-class TP/FP/FN/TN are counted over the `(y_true, y_pred)` pairs; precision, recall, and F1 are derived from those counts.
4. Macro F1 is the simple unweighted mean of the three per-class F1 scores (two for strategy A).
5. The resulting flat dict is appended as a new row; existing rows are never modified.

---

## Statistical Evaluation

`statisticalTests_evaluation.py` runs pairwise two-proportion z-tests on accuracy across runs stored in `ResultOverview_allTests.csv`.

```bash
python statisticalTests_evaluation.py [path/to/ResultOverview_allTests.csv]
```

Defaults to `ResultOverview_allTests.csv` next to the script. Outputs are written to `eval_output/`:

| File | Description |
|------|-------------|
| `runs_annotated.csv` | Input data with added `ds`, `n_eff`, and `n_correct` columns |
| `pairwise_ztests_accuracy.csv` | All z-test results, one row per comparison |
| `evaluation_report.md` | Human-readable Markdown report with tables and plain-language summary |

### How rows are matched

**Row order does not matter.** Each comparison looks up rows by four key columns:

| Column | Role |
|--------|------|
| `strategy` | `A`, `B`, `C`, or `D` |
| `few_shot_n` | Number of few-shot examples (`0` = zero-shot) |
| `batch_size` | Pairs per API call |
| `ds` | Derived from `dataset_file`: prefix `NR_` → `NR_Web`, otherwise → `Web` |

If no matching row is found a warning is printed and that comparison is skipped (no crash). If multiple rows match, the first is used with a warning.

### Required columns

| Always needed | Strategy-specific (for `n_eff`) |
|---|---|
| `strategy`, `few_shot_n`, `dataset_file`, `batch_size`, `accuracy` | A: `true_positive_attack`, `false_positive_attack`, `false_negative_attack`, `true_negative_attack` |
| | B: `predicted_count_attack`, `predicted_count_support` |
| | C / D: `predicted_count_attack`, `predicted_count_support`, `predicted_count_no_relation` |

### Adding new results

New rows appended to `ResultOverview_allTests.csv` are **automatically picked up** on the next run — no script changes needed. However, a new run is only **evaluated** if there is a hardcoded comparison in `run_accuracy_tests()` that references its `(strategy, few_shot_n, ds, batch_size)` combination. Runs that don't match any comparison are silently ignored. To include a new comparison, add it to the relevant group block in `run_accuracy_tests()`.

### Comparison groups

| Group | What is compared |
|-------|-----------------|
| 1 — Zero-shot vs few-shot | Same strategy and batch size (b10); varies `few_shot_n` |
| 1b — Few-shot dose-response | All pairwise `few_shot_n` combinations within a strategy |
| 2 — Batch size | Same strategy, zero-shot; varies `batch_size` |
| 3 — Cross-strategy (zero-shot) | Different strategies at matched batch size (note: A/B use `Web`, C uses `NR_Web`) |
| 3b — Cross-strategy (few-shot) | A vs B at matched `few_shot_n` (batch 10, `Web`) |
| 4 — C vs D | Both zero-shot, batch 10, `NR_Web` |

---

## Evaluation

Results are printed to the console after each run. For each strategy, the output includes:

- Ground-truth and predicted label distributions
- Confusion matrix
- Per-class TP/FP/FN/TN, precision, recall, and F1 (with derivation shown)
- Macro F1
- Accuracy

You can also run `evaluation.py` standalone on any existing results CSV:

```bash
python evaluation.py [path/to/results2.csv]
```

**Evaluation logic by strategy:**

| Strategy | Correct when |
|----------|-------------|
| A | `(support==0) ↔ (pred_attack==1)` |
| B | `support==0 → pred_attack==1`; `support==1 → pred_support==1`; `support==2` → always wrong |
| C / D | The flag matching the ground-truth class is 1 |

---
