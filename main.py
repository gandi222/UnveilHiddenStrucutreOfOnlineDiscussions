import logging
from pathlib import Path

from langchain_ollama import ChatOllama

from pipeline import load_arrow_dataset, run_evaluation
from prompts import STRATEGIES

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "results2.csv"
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
MODEL = "gemma4:latest"
LIMIT = 20          # set to None to use all 4000 pairs
BATCH_SIZE = 10
DELAY_SECONDS = 0
MAX_RETRIES = 1

# "A" — binary, "B" — two-class, "C" — three-class, or all at once
STRATEGIES_TO_RUN = ["C"]
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    Path(OUTPUT_CSV).unlink(missing_ok=True)
    # log.info("Cleared %s (fresh run)", OUTPUT_CSV)

    df = load_arrow_dataset(ARROW_FILE, limit=LIMIT)
    client = ChatOllama(base_url=BASE_URL, model=MODEL, temperature=1.0)

    invalid = [s for s in STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    for strategy in STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)
        run_evaluation(df, strategy, client, BATCH_SIZE, DELAY_SECONDS, MAX_RETRIES, OUTPUT_CSV)

    log.info("Done. Run evaluation.py for full metrics.")


if __name__ == "__main__":
    main()
