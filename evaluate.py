"""
LLM-based evaluation of argument relationships using the Gemini API.

Each argument pair (arg1, arg2) is sent to the model individually.
Three prompt strategies are tested to compare how prompt design affects accuracy:
  A: Binary      — only asks whether arg2 attacks arg1 (attack: 0/1)
  B: Two-class   — asks to choose between attack or support (each 0/1)
  C: Three-class — asks to choose between attack, support, or neither (each 0/1)

Usage:
  1. Set GEMINI_API_KEY as an environment variable
  2. pip install -r requirements.txt
  3. python evaluate.py

Set LIMIT (below) to a small int for a quick smoke-test.
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
OUTPUT_CSV = "results.csv"                     # where results are saved
MODEL = "gemini-2.0-flash"                     # Gemini model to use
LIMIT = 1         # max argument pairs to process; set to None to use all 1284
DELAY_SECONDS = 4 # seconds to wait between API calls (Free Tier allows ~15 req/min)
MAX_RETRIES = 1   # how many times to retry a failed API call before skipping the row
# ---------------------------------------------------------------------------

# Set up logging so progress and warnings are printed to the terminal with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# Each function takes one argument pair and returns a complete prompt string.
# ---------------------------------------------------------------------------

def _base(arg1: str, arg2: str) -> str:
    # Shared header used by all three strategies
    return (
        f"Given these two arguments:\n"
        f"Arg1: {arg1}\n"
        f"Arg2: {arg2}\n\n"
    )


def prompt_a(arg1: str, arg2: str) -> str:
    # Strategy A: simplest framing — binary yes/no for "attack"
    return (
        _base(arg1, arg2)
        + "Does Arg2 attack Arg1?\n"
        + 'Respond with ONLY a JSON object, no explanation: {"attack": 0 or 1}'
    )


def prompt_b(arg1: str, arg2: str) -> str:
    # Strategy B: two competing labels — forces the model to choose attack vs. support
    return (
        _base(arg1, arg2)
        + "Classify the relationship between Arg2 and Arg1.\n"
        + "Exactly one of the following should be 1:\n"
        + "  attack  — Arg2 attacks/undermines Arg1\n"
        + "  support — Arg2 supports/reinforces Arg1\n"
        + 'Respond with ONLY a JSON object, no explanation: {"attack": 0 or 1, "support": 0 or 1}'
    )


def prompt_c(arg1: str, arg2: str) -> str:
    # Strategy C: three labels — adds "neither" as an escape hatch for unrelated pairs
    return (
        _base(arg1, arg2)
        + "Classify the relationship between Arg2 and Arg1.\n"
        + "Exactly one of the following should be 1:\n"
        + "  attack  — Arg2 attacks/undermines Arg1\n"
        + "  support — Arg2 supports/reinforces Arg1\n"
        + "  neither — Arg2 has no clear argumentative relation to Arg1\n"
        + 'Respond with ONLY a JSON object, no explanation: '
        + '{"attack": 0 or 1, "support": 0 or 1, "neither": 0 or 1}'
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


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from the model response.
    Strips markdown code fences (```json ... ```) that the model sometimes adds."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text!r}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Evaluation loop — one API call per argument pair
# ---------------------------------------------------------------------------

def run_evaluation(df: pd.DataFrame, strategy: str, client: genai.Client) -> list[dict]:
    """Iterate over all argument pairs, call the API once per pair, and collect results."""
    prompt_fn = STRATEGIES[strategy]
    results = []
    total = len(df)

    log.info("Starting strategy %s (%d pairs)", strategy, total)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        prompt = prompt_fn(row.arg1, row.arg2)

        # Default all prediction fields to None; filled in if the API call succeeds
        # Prefixed with "pred_" to distinguish from the "support" ground-truth column
        labels = {"pred_attack": None, "pred_support": None, "pred_neither": None}

        try:
            raw = _call_api(client, prompt)
            parsed = _parse_json(raw)
            # Map model keys (attack/support/neither) to pred_ prefixed columns
            for key in ("attack", "support", "neither"):
                if key in parsed:
                    labels[f"pred_{key}"] = int(parsed[key])
        except Exception as exc:
            # Log the failure but continue — the row is saved with None labels
            log.warning("Row %d/%d strategy %s failed: %s", idx, total, strategy, exc)

        results.append({
            "arg1": row.arg1,
            "arg2": row.arg2,
            "support": row.support,  # original ground-truth label from the dataset
            "strategy": strategy,
            "pred_attack": labels["pred_attack"],
            "pred_support": labels["pred_support"],
            "pred_neither": labels["pred_neither"],
        })

        # Progress log every 10 rows so long runs are visible in the terminal
        if idx % 10 == 0:
            log.info("  strategy %s: %d/%d done", strategy, idx, total)

        # Pause between calls to stay within the Free Tier rate limit
        time.sleep(DELAY_SECONDS)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Read the API key from the environment (set with: export GEMINI_API_KEY="...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it in your terminal before running."
        )

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
    log.info("Value counts for 'support':\n%s", df["support"].value_counts().to_string())
    log.info("Sample row:\n%s", df.iloc[0].to_string())
    log.info("============================")

    # Shuffle rows so any LIMIT slice is a random sample, not just the first N rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if LIMIT is not None:
        df = df.iloc[:LIMIT]
        log.info("Limited to %d pairs", LIMIT)

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)

    client = genai.Client(api_key=api_key)

    # Run all three strategies in sequence; each produces one result row per pair
    all_results = []
    for strategy in STRATEGIES:
        results = run_evaluation(df, strategy, client)
        all_results.extend(results)

    # Combine into a single DataFrame and save — each pair appears 3 times (once per strategy)
    out_df = pd.DataFrame(all_results, columns=[
        "arg1", "arg2", "support", "strategy", "pred_attack", "pred_support", "pred_neither"
    ])
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Results saved to %s (%d rows)", OUTPUT_CSV, len(out_df))


if __name__ == "__main__":
    main()
