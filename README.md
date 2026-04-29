# Argument Relation Evaluation with Gemini

LLM-based evaluation of argument pair relationships using the Google Gemini API. Multiple argument pairs are packed into a single prompt (batching) to reduce API calls and token cost. Three prompt strategies — binary, two-class, and three-class — are evaluated independently against a ground-truth dataset.

---

## Overview

Given a dataset of argument pairs `(arg1, arg2)`, the system asks a Gemini model to classify the relation between them. Three increasingly granular strategies are tested:

| Strategy | Classes predicted | Output fields |
|----------|-------------------|---------------|
| A — Binary | Attack only | `pred_attack` |
| B — Two-class | Attack or Support | `pred_attack`, `pred_support` |
| C — Three-class | Attack, Support, or No Relation | `pred_attack`, `pred_support`, `pred_neither` |

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

The dataset contains **1284 argument pairs** total. Rows are shuffled before processing so any `LIMIT` slice is a random sample.

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
google-genai
pyarrow
pandas
```

### 2. Set your Gemini API key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run

```bash
python evaluate2.py
```

---

## Configuration

All tunable variables are at the top of [evaluate2.py](evaluate2.py):

```python
ARROW_FILE       = "NR_WebDataset/data-00000-of-00001.arrow"  # input dataset path
OUTPUT_CSV       = "results2.csv"          # output file
MODEL            = "gemini-2.5-flash-lite" # Gemini model to use
LIMIT            = 10     # max pairs to process; None = all 1284
BATCH_SIZE       = 5      # pairs per API call (higher = fewer calls, more tokens per call)
DELAY_SECONDS    = 10     # pause between API calls (Free Tier: ~15 req/min)
MAX_RETRIES      = 1      # retry attempts per failed API call

STRATEGIES_TO_RUN = ["A", "B", "C"]  # which strategies to run
```

**Rate limit guidance:** The Gemini Free Tier allows ~15 requests/minute. With `BATCH_SIZE=5` and `DELAY_SECONDS=10`, each batch takes ~10 s → ~6 req/min, well within limits. Increase `BATCH_SIZE` or reduce `DELAY_SECONDS` on a paid plan.

---

## Prompt Strategies

Each strategy packs `BATCH_SIZE` argument pairs into one prompt and asks the model to return a JSON array with one result per pair.

### Strategy A — Binary

```
In this task, you will be given two arguments and your goal is to classify
whether Arg2 attacks Arg1 based on the definition below.
'Attack': Arg2 contradicts or opposes Arg1.

For each pair, output 1 if Arg2 attacks Arg1, or 0 if it does not.

Pair 1:
Arg1: <arg1>
Arg2: <arg2>

...

Respond with ONLY a JSON array of exactly N objects, one per pair, in order.
Each object must be either {"attack": 0} or {"attack": 1}.
```

### Strategy B — Two-class

```
In this task, you will be given two arguments and your goal is to classify
the relation between them as either "Support" or "Attack" based on the definitions below.
'Support': Arg2 is in favour of or agrees with Arg1.
'Attack': Arg2 contradicts or opposes Arg1.

For each pair, set the matching field to 1 and the other to 0.

Pair 1:
Arg1: <arg1>
Arg2: <arg2>

...

Respond with ONLY a JSON array of exactly N objects, one per pair, in order.
Each object must follow this schema: {"attack": 0, "support": 0}
Exactly one field per object must be 1.
```

### Strategy C — Three-class

```
In this task, you will be given two arguments and your goal is to classify
the relation between them as either "Support", "Attack", or "No Relation" based on the definitions below.
'Support': Arg2 is in favour of or agrees with Arg1.
'Attack': Arg2 contradicts or opposes Arg1.
'No Relation': Arg2 has no meaningful relation to Arg1.

For each pair, set the matching field to 1 and all others to 0.

Pair 1:
Arg1: <arg1>
Arg2: <arg2>

...

Respond with ONLY a JSON array of exactly N objects, one per pair, in order.
Each object must follow this schema: {"attack": 0, "support": 0, "neither": 0}
Exactly one field per object must be 1.
```

**Prompt design rationale:**
- Explicit label definitions reduce model ambiguity
- The exact pair count `N` is included so the model doesn't need to count
- Both valid JSON structures are shown to prevent pattern-matching on a single example
- "ONLY" and "Exactly one" instructions constrain hallucinated formats

---

## Output

Results are written to `results2.csv`. Each argument pair appears once per strategy run.

| Column | Description |
|--------|-------------|
| `arg1` | First argument text |
| `arg2` | Second argument text |
| `support` | Ground-truth label (0=Attack, 1=Support, 2=No Relation) |
| `support_label` | Human-readable ground-truth (Attack / Support / No Relation) |
| `strategy` | Which strategy produced this row (A, B, or C) |
| `pred_attack` | Model prediction: 1 if attack, 0 if not, `<NA>` if batch failed |
| `pred_support` | Model prediction: 1 if support, 0 if not, `<NA>` if batch failed |
| `pred_neither` | Model prediction: 1 if no relation, 0 if not, `<NA>` if batch failed |

If a batch API call fails after `MAX_RETRIES` attempts, all rows in that batch receive `<NA>` for prediction columns instead of crashing the run.

---

## Project Structure

```
.
├── evaluate2.py              # main evaluation script
├── requirements.txt          # Python dependencies
├── results2.csv              # output (generated on run)
└── NR_WebDataset/
    ├── data-00000-of-00001.arrow   # argument pair dataset (Arrow IPC format)
    ├── dataset_info.json           # schema metadata
    └── state.json                  # HuggingFace dataset state
```

---

