"""
LLM-based evaluation of argument relationships using the Gemini API.
Batched version: multiple argument pairs are packed into one prompt to save API tokens.

Instead of one API call per pair, BATCH_SIZE pairs are sent together.
The model returns a JSON array with one result per pair in the same order.

Three prompt strategies are tested:
  A: Binary      — only asks whether arg2 attacks arg1 (attack: 0/1)
  B: Two-class   — asks to choose between attack or support (each 0/1)
  C: Three-class — asks to choose between attack, support, or neither (each 0/1)

Usage:
  1. Set GEMINI_API_KEY as an environment variable
  2. pip install -r requirements.txt
  3. python evaluate2.py
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc  # Arrow IPC stream format (used by HuggingFace datasets)
from google import genai

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "data-00000-of-00001 (2).arrow"  # input dataset
OUTPUT_CSV = "results2.csv"                    # where results are saved
MODEL = "gemini-2.5-flash-lite"                # Gemini model to use
LIMIT = 5        # max argument pairs to process; set to None to use all 1284
BATCH_SIZE = 5    # how many pairs to pack into a single API call (higher = fewer calls)
DELAY_SECONDS = 10 # seconds to wait between API calls (Free Tier allows ~15 req/min)
MAX_RETRIES = 1   # how many times to retry a failed API call before skipping the batch

# Which prompt strategies to run — add or remove entries to enable/disable each one:
#   "A" — binary:      only predicts attack            (pred_attack)
#   "B" — two-class:   predicts attack or support      (pred_attack, pred_support)
#   "C" — three-class: predicts attack, support, neither (pred_attack, pred_support, pred_neither)
#   "D" — still needs to be implemented for QBAF
# Each enabled strategy runs independently; the output contains one row per pair per strategy.
# Ground-truth label meaning in the dataset (column "support"):
#   0 = Attack  |  1 = Support  |  2 = No Relation
STRATEGIES_TO_RUN = ["A", "B", "C"]  # e.g. ["A"] or ["B", "C"] or ["A", "B", "C"]
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
    # Strategy A: for each pair, only ask whether arg2 attacks arg1
    return (
        f"For each of the following argument pairs, does Arg2 attack Arg1?\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array with one object per pair, in order.\n"
        f"Use only the integers 0 or 1 (no decimals, no strings).\n"
        f'[{{"attack": 1}}, {{"attack": 0}}, ...]'
    )


def prompt_b(pairs: list[tuple[str, str]]) -> str:
    # Strategy B: for each pair, choose between attack or support (exactly one = 1)
    return (
        f"Classify the relationship between Arg2 and Arg1 for each pair.\n"
        f"Exactly one field per pair must be 1; the other must be 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array with one object per pair, in order.\n"
        f"Use only the integers 0 or 1 (no decimals, no strings).\n"
        f'[{{"attack": 1, "support": 0}}, {{"attack": 0, "support": 1}}, ...]'
    )


def prompt_c(pairs: list[tuple[str, str]]) -> str:
    # Strategy C: for each pair, choose between attack, support, or neither (exactly one = 1)
    return (
        f"Classify the relationship between Arg2 and Arg1 for each pair.\n"
        f"Exactly one field per pair must be 1; the others must be 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array with one object per pair, in order.\n"
        f"Use only the integers 0 or 1 (no decimals, no strings).\n"
        f'[{{"attack": 1, "support": 0, "neither": 0}}, {{"attack": 0, "support": 0, "neither": 1}}, ...]'
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

def _call_api(client: genai.Client, prompt: str) -> str:
    """Send a prompt to Gemini and return the raw response text.
    Retries up to MAX_RETRIES times on any error, then raises."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
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

def run_evaluation(df: pd.DataFrame, strategy: str, client: genai.Client) -> list[dict]:
    """Split the dataset into batches, call the API once per batch, and collect results."""
    prompt_fn = STRATEGIES[strategy]
    results = []
    total = len(df)
    rows = list(df.itertuples(index=False))  # convert to list for easy slicing

    log.info("Starting strategy %s (%d pairs, batch size %d)", strategy, total, BATCH_SIZE)

    # Step through the dataset in chunks of BATCH_SIZE
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]
        pairs = [(r.arg1, r.arg2) for r in batch_rows]

        # Build one prompt that contains all pairs in this batch
        prompt = prompt_fn(pairs)
        log.info("--- Prompt (batch %d-%d) ---\n%s\n---", batch_start + 1, batch_start + len(batch_rows), prompt)
        parsed_batch = None  # will hold the list of label dicts if the call succeeds

        try:
            raw = _call_api(client, prompt)
            # Parse the JSON array; must have exactly one entry per pair
            parsed_batch = _parse_json_array(raw, len(batch_rows))
        except Exception as exc:
            # If the batch call fails, all rows in it will have None labels
            log.warning(
                "Batch %d-%d strategy %s failed: %s",
                batch_start + 1, batch_start + len(batch_rows), strategy, exc,
            )

        # Map each parsed result back to its original row
        for i, row in enumerate(batch_rows):
            # Prefixed with "pred_" to distinguish from the "support" ground-truth column
            labels = {"pred_attack": None, "pred_support": None, "pred_neither": None}
            if parsed_batch is not None:
                item = parsed_batch[i]  # the model's answer for this specific pair
                for key in ("attack", "support", "neither"):
                    if key in item:
                        labels[f"pred_{key}"] = int(float(item[key]))

            results.append({
                "arg1": row.arg1,
                "arg2": row.arg2,
                "support": row.support,  # original ground-truth label from the dataset
                "strategy": strategy,
                "pred_attack": labels["pred_attack"],
                "pred_support": labels["pred_support"],
                "pred_neither": labels["pred_neither"],
            })

        # Progress log at every 50 pairs and at the very end
        done = min(batch_start + BATCH_SIZE, total)
        if done % 50 == 0 or done == total:
            log.info("  strategy %s: %d/%d done", strategy, done, total)

        # Pause between batch calls to stay within the Free Tier rate limit
        time.sleep(DELAY_SECONDS)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Read the API key from the environment (set with: export GEMINI_API_KEY="...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set. Export it in your terminal before running.")

    # Load the Arrow dataset (HuggingFace IPC stream format)
    arrow_path = Path(ARROW_FILE)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with ipc.open_stream(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()
    # "support" is the original column name from the dataset (1 = support, 0 = attack)

    # Print a summary of the input file so the user can verify it loaded correctly
    log.info("=== Input file structure ===")
    log.info("Rows:    %d", len(df))
    log.info("Columns: %s", list(df.columns))
    log.info("Dtypes:\n%s", df.dtypes.to_string())
    log.info("Value counts for 'support' (0=Attack, 1=Support, 2=No Relation):\n%s", df["support"].value_counts().to_string())
    log.info("Sample row:\n%s", df.iloc[0].to_string())
    log.info("============================")

    # Shuffle rows so any LIMIT slice is a random sample, not just the first N rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if LIMIT is not None:
        df = df.iloc[:LIMIT]
        log.info("Limited to %d pairs", LIMIT)

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)

    client = genai.Client(api_key=api_key)

    # Validate the selected strategies
    invalid = [s for s in STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    all_results = []
    for strategy in STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)
        all_results.extend(run_evaluation(df, strategy, client))

    out_df = pd.DataFrame(all_results, columns=[
        "arg1", "arg2", "support", "strategy", "pred_attack", "pred_support", "pred_neither"
    ])

    # Add a human-readable version of the ground-truth label next to the numeric column
    support_map = {0: "Attack", 1: "Support", 2: "No Relation"}
    out_df.insert(
        out_df.columns.get_loc("support") + 1,
        "support_label",
        out_df["support"].map(support_map),
    )

    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Results saved to %s (%d rows)", OUTPUT_CSV, len(out_df))


if __name__ == "__main__":
    main()
