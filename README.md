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

One row appended per strategy run (when `TRACK_RESULTS=1`). Columns include run metadata (timestamp, git commit, model, config), ground-truth and predicted class counts, per-class confusion matrix values (TP/FP/FN/TN), per-class precision/recall/F1, and macro F1.

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
