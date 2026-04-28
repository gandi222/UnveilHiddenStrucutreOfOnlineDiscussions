"""
LLM-based evaluation of argument relationships using the Gemini API.

Three prompt strategies are tested:
  A: Binary  — attack only (0/1)
  B: Two-class — attack + support (each 0/1)
  C: Three-class — attack + support + neither (each 0/1)

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
import pyarrow.ipc as ipc
from google import genai

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "data-00000-of-00001 (2).arrow"
OUTPUT_CSV = "results.csv"
MODEL = "gemini-2.0-flash"
LIMIT = 1             # int to cap the number of argument pairs; None = all
DELAY_SECONDS = 4     # sleep between API calls (Free Tier: ~15 RPM)
MAX_RETRIES = 1       # retries on rate-limit / transient errors
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _base(arg1: str, arg2: str) -> str:
    return (
        f"Given these two arguments:\n"
        f"Arg1: {arg1}\n"
        f"Arg2: {arg2}\n\n"
    )


def prompt_a(arg1: str, arg2: str) -> str:
    return (
        _base(arg1, arg2)
        + "Does Arg2 attack Arg1?\n"
        + 'Respond with ONLY a JSON object, no explanation: {"attack": 0 or 1}'
    )


def prompt_b(arg1: str, arg2: str) -> str:
    return (
        _base(arg1, arg2)
        + "Classify the relationship between Arg2 and Arg1.\n"
        + "Exactly one of the following should be 1:\n"
        + "  attack  — Arg2 attacks/undermines Arg1\n"
        + "  support — Arg2 supports/reinforces Arg1\n"
        + 'Respond with ONLY a JSON object, no explanation: {"attack": 0 or 1, "support": 0 or 1}'
    )


def prompt_c(arg1: str, arg2: str) -> str:
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


STRATEGIES = {
    "A": prompt_a,
    "B": prompt_b,
    "C": prompt_c,
}


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def _call_api(client: genai.Client, prompt: str) -> str:
    """Call Gemini and return raw response text, retrying up to MAX_RETRIES times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except Exception as exc:
            log.warning("API call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(5)
            else:
                raise RuntimeError("Max retries exceeded") from exc


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a response string."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text!r}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(df: pd.DataFrame, strategy: str, client: genai.Client) -> list[dict]:
    prompt_fn = STRATEGIES[strategy]
    results = []
    total = len(df)

    log.info("Starting strategy %s (%d pairs)", strategy, total)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        prompt = prompt_fn(row.arg1, row.arg2)
        labels = {"attack": None, "support": None, "neither": None}

        try:
            raw = _call_api(client, prompt)
            parsed = _parse_json(raw)
            for key in labels:
                if key in parsed:
                    labels[key] = int(parsed[key])
        except Exception as exc:
            log.warning("Row %d/%d strategy %s failed: %s", idx, total, strategy, exc)

        results.append({
            "arg1": row.arg1,
            "arg2": row.arg2,
            "groundtruth": row.groundtruth,
            "strategy": strategy,
            "attack": labels["attack"],
            "support": labels["support"],
            "neither": labels["neither"],
        })

        if idx % 10 == 0:
            log.info("  strategy %s: %d/%d done", strategy, idx, total)

        time.sleep(DELAY_SECONDS)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Copy .env.example → .env and fill in your key."
        )

    # Load dataset
    arrow_path = Path(ARROW_FILE)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with ipc.open_stream(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()
    df = df.rename(columns={"support": "groundtruth"})

    # Shuffle and optionally limit
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if LIMIT is not None:
        df = df.iloc[:LIMIT]
        log.info("Limited to %d pairs", LIMIT)

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)

    client = genai.Client(api_key=api_key)

    all_results = []
    for strategy in STRATEGIES:
        results = run_evaluation(df, strategy, client)
        all_results.extend(results)

    out_df = pd.DataFrame(all_results, columns=[
        "arg1", "arg2", "groundtruth", "strategy", "attack", "support", "neither"
    ])
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Results saved to %s (%d rows)", OUTPUT_CSV, len(out_df))


if __name__ == "__main__":
    main()
