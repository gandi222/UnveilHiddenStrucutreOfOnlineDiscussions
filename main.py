"""
LLM-based evaluation of argument relationships using a locally-hosted Ollama model.
Batched version: multiple argument pairs are packed into one prompt to save tokens.

Instead of one API call per pair, BATCH_SIZE pairs are sent together.
The model returns a JSON array with one result per pair in the same order.

Three prompt strategies are tested:
  A: Binary      — only asks whether arg2 attacks arg1 (attack: 0/1)
  B: Two-class   — asks to choose between attack or support (each 0/1)
  C: Three-class — asks to choose between attack, support, or neither (each 0/1)

Usage:
  1. pip install langchain-ollama
  2. python evaluate2.py
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc  # Arrow IPC stream format (used by HuggingFace datasets)
from langchain_ollama import ChatOllama

from evaluation import SUPPORT_MAP, add_correct_column, accuracy_summary

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"  # input dataset
OUTPUT_CSV = "results2.csv"                    # where results are saved
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"  # WU Ollama server
MODEL = "gemma4:latest"                        # model available on the Ollama server
LIMIT = 10       # max argument pairs to process; set to None to use all 1284
BATCH_SIZE = 5    # how many pairs to pack into a single API call (higher = fewer calls)
DELAY_SECONDS = 1  # seconds to wait between API calls
MAX_RETRIES = 1   # how many times to retry a failed API call before skipping the batch

# Which prompt strategies to run — add or remove entries to enable/disable each one:
#   "A" — binary:      only predicts attack            (pred_attack)
#   "B" — two-class:   predicts attack or support      (pred_attack, pred_support)
#   "C" — three-class: predicts attack, support, neither (pred_attack, pred_support, pred_neither)
#   "D" — still needs to be implemented for QBAF
# Each enabled strategy runs independently; the output contains one row per pair per strategy.
# Ground-truth label meaning in the dataset (column "support"):
#   0 = Attack  |  1 = Support  |  2 = No Relation
STRATEGIES_TO_RUN = ["A"]  # e.g. ["A"] or ["B", "C"] or ["A", "B", "C"]
# ---------------------------------------------------------------------------

# Set up logging so progress and warnings are printed to the terminal with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders (batched)
# Each function takes a list of (arg1, arg2) tuples and returns one prompt string
# that asks the model to classify all pairs at once.
# ---------------------------------------------------------------------------

def _format_pairs(pairs: list[tuple[str, str]]) -> str:
    # Format all pairs as a numbered list so the model can reference them by index
    lines = []
    for i, (arg1, arg2) in enumerate(pairs, start=1):
        lines.append(f"Pair {i}:\nArg1: {arg1}\nArg2: {arg2}")
    return "\n\n".join(lines)


def prompt_a(pairs: list[tuple[str, str]]) -> str:
    # Strategy A: binary — only predicts whether arg2 attacks arg1.
    # Prompt design: explicit label definition reduces ambiguity; exact count {n} tells
    # the model how many objects to output so it doesn't have to count pairs itself;
    # both valid JSON objects are shown so the model doesn't pattern-match on a single example.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"whether Arg2 attacks Arg1 based on the definition below.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, output 1 if Arg2 attacks Arg1, or 0 if it does not.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must be either {{\"attack\": 0}} or {{\"attack\": 1}}.\n"
    )


def prompt_b(pairs: list[tuple[str, str]]) -> str:
    # Strategy B: two-class — predicts attack or support.
    # Schema shows all zeros as a key-structure template; the "exactly one must be 1"
    # instruction tells the model what to fill in, avoiding confusion from a fixed example.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\" or \"Attack\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, set the matching field to 1 and the other to 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


def prompt_c(pairs: list[tuple[str, str]]) -> str:
    # Strategy C: three-class — predicts attack, support, or neither.
    # Same schema/count design as B; adds "No Relation" class with its own definition.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\", \"Attack\", or \"No Relation\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n"
        f"'No Relation': Arg2 has no meaningful relation to Arg1.\n\n"
        f"For each pair, set the matching field to 1 and all others to 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0, \"neither\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


# Maps strategy name → prompt builder function
STRATEGIES = {
    "A": prompt_a,
    "B": prompt_b,
    "C": prompt_c,
}


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def _call_api(llm: ChatOllama, prompt: str) -> str:
    """Send a prompt to the Ollama server and return the raw response text.
    Retries up to MAX_RETRIES times on any error, then raises."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as exc:
            log.warning("API call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(5)  # short pause before retrying
            else:
                raise RuntimeError("Max retries exceeded") from exc


def _parse_json_array(text: str, expected: int) -> list[dict]:
    """Extract a JSON array from the model response and verify it has the right length.
    Strips markdown code fences (```json ... ```) that the model sometimes adds.
    Raises if the array length doesn't match the number of pairs in the batch."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text!r}")
    result = json.loads(match.group())
    if len(result) != expected:
        raise ValueError(f"Expected {expected} items in response, got {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Evaluation loop — one API call per batch of BATCH_SIZE pairs
# ---------------------------------------------------------------------------

def run_evaluation(df: pd.DataFrame, strategy: str, client: ChatOllama) -> list[dict]:
    """Split the dataset into batches, call the API once per batch, and collect results."""
    prompt_fn = STRATEGIES[strategy]
    results = []
    total = len(df)
    rows = list(df.itertuples(index=False))  # convert to list for easy slicing

    log.info("=" * 60)
    log.info("STRATEGY %s  |  %d pairs  |  batch size %d", strategy, total, BATCH_SIZE)
    log.info("=" * 60)

    # Step through the dataset in chunks of BATCH_SIZE
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch_rows)
        pairs = [(r.arg1, r.arg2) for r in batch_rows]

        # Build one prompt that contains all pairs in this batch
        prompt = prompt_fn(pairs)

        log.info("-" * 60)
        log.info("BATCH %d–%d  (strategy %s)", batch_start + 1, batch_end, strategy)
        log.info("-" * 60)
        log.info("[PROMPT]\n%s", prompt)

        parsed_batch = None  # will hold the list of label dicts if the call succeeds

        try:
            raw = _call_api(client, prompt)
            log.info("[LLM RESPONSE]\n%s", raw)
            # Parse the JSON array; must have exactly one entry per pair
            parsed_batch = _parse_json_array(raw, len(batch_rows))
            log.info("[PARSED]  %s", parsed_batch)
        except Exception as exc:
            # If the batch call fails, all rows in it will have None labels
            log.warning("[FAILED]  Batch %d–%d strategy %s: %s", batch_start + 1, batch_end, strategy, exc)

        # Map each parsed result back to its original row
        for i, row in enumerate(batch_rows):
            # Prefixed with "pred_" to distinguish from the "support" ground-truth column
            labels = {"pred_attack": None, "pred_support": None, "pred_neither": None}
            if parsed_batch is not None:
                item = parsed_batch[i]  # the model's answer for this specific pair
                for key in ("attack", "support", "neither"):
                    if key in item:
                        # int(float(...)) handles model returning "1.0" (string) or 1.0 (float)
                        labels[f"pred_{key}"] = int(float(item[key]))

            results.append({
                "orig_idx": row.orig_idx,  # arrow file row index — ties support to its source row
                "arg1": row.arg1,
                "arg2": row.arg2,
                "support": row.support,  # original ground-truth label from the dataset
                "strategy": strategy,
                "pred_attack": labels["pred_attack"],
                "pred_support": labels["pred_support"],
                "pred_neither": labels["pred_neither"],
            })

        log.info("[PROGRESS]  strategy %s: %d/%d pairs done", strategy, batch_end, total)

        # Pause between batch calls to stay within the Free Tier rate limit
        time.sleep(DELAY_SECONDS)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Delete stale output so a crash mid-run doesn't leave partial results from a previous run.
    Path(OUTPUT_CSV).unlink(missing_ok=True)
    log.info("Cleared %s (fresh run)", OUTPUT_CSV)

    # Load the Arrow dataset (HuggingFace IPC stream format)
    arrow_path = Path(ARROW_FILE)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with ipc.open_stream(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()
    # "support" is the original column name from the dataset (1 = support, 0 = attack)

    # Preserve the original row index from the arrow file before any shuffling.
    # This is the ground-truth anchor: even after randomization the correct support
    # value can be verified by looking up orig_idx in the source file.
    df["orig_idx"] = df.index

    # Print a summary of the input file so the user can verify it loaded correctly
    log.info("=== Input file structure ===")
    log.info("Rows:    %d", len(df))
    log.info("Columns: %s", list(df.columns))
    log.info("Dtypes:\n%s", df.dtypes.to_string())
    log.info("Value counts for 'support' (0=Attack, 1=Support, 2=No Relation):\n%s", df["support"].value_counts().to_string())
    log.info("Sample row:\n%s", df.iloc[0].to_string())
    log.info("============================")

    # Shuffle rows so any LIMIT slice is a random sample, not just the first N rows
    df = df.sample(frac=1).reset_index(drop=True)
    if LIMIT is not None:
        df = df.iloc[:LIMIT]
        log.info("Limited to %d pairs", LIMIT)

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)

    client = ChatOllama(base_url=BASE_URL, model=MODEL, temperature=1.0)

    # Validate the selected strategies
    invalid = [s for s in STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    all_results = []
    for strategy in STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)
        all_results.extend(run_evaluation(df, strategy, client))

    out_df = pd.DataFrame(all_results, columns=[
        "orig_idx", "arg1", "arg2", "support", "strategy",
        "pred_attack", "pred_support", "pred_neither",
    ])

    # Cast numeric columns to integer types.
    # pred_* columns use nullable Int64 so that failed-batch None values stay as <NA>
    # instead of forcing the whole column to float64 (which would show 1 as 1.0).
    out_df["support"] = out_df["support"].astype(int)
    for col in ("pred_attack", "pred_support", "pred_neither"):
        out_df[col] = out_df[col].astype("Int64")

    # Add a human-readable version of the ground-truth label next to the numeric column
    out_df.insert(
        out_df.columns.get_loc("support") + 1,
        "support_label",
        out_df["support"].map(SUPPORT_MAP),
    )

    add_correct_column(out_df)

    out_df = out_df.rename(columns={"support": "support [true value]"})

    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Results saved to %s (%d rows)", OUTPUT_CSV, len(out_df))

    # Print per-strategy accuracy summary
    log.info("=== Accuracy summary ===")
    for strat, acc in accuracy_summary(out_df).items():
        grp = out_df[out_df["strategy"] == strat]["correct"].dropna()
        log.info("Strategy %s: %d/%d correct (%.1f%%)", strat, grp.sum(), len(grp), 100 * acc)
    log.info("========================")


if __name__ == "__main__":
    main()
