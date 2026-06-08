import logging
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama

from evaluation import append_to_overview, print_all_results
from pipeline import load_arrow_dataset, run_evaluation, sample_few_shot
from prompts import STRATEGIES

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "results2.csv"
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
MODEL = "gemma4:latest"
LIMIT = 1000          # set to None to use all 4000 pairs
BATCH_SIZE = 20
DELAY_SECONDS = 0
MAX_RETRIES = 2

# "A" — binary, "B" — two-class, "C" — three-class, "D" — three-class + relevance score (zero-shot if dataset has no relevance column), or all at once
STRATEGIES_TO_RUN = ["A","B"]

# Number of labeled examples injected into each prompt as few-shot context.
# 0 = no few-shot (zero-shot). Sampled once per strategy, balanced across classes.
# For strategy B only Attack/Support examples are eligible (not No Relation).
FEW_SHOT_N = 0
TRACK_RESULTS = 1   # set to 0 to skip writing ResultOverview_allTests.csv
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("Date: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info(
        "Configuration — model: %s | strategies: %s | limit: %s | few_shot_n: %d | "
        "batch_size: %d | delay: %ds | max_retries: %d | output: %s",
        MODEL, STRATEGIES_TO_RUN, LIMIT, FEW_SHOT_N,
        BATCH_SIZE, DELAY_SECONDS, MAX_RETRIES, OUTPUT_CSV,
    )

    Path(OUTPUT_CSV).unlink(missing_ok=True)

    full_df = load_arrow_dataset(ARROW_FILE)  # load all rows; LIMIT applied per strategy below
    client = ChatOllama(base_url=BASE_URL, model=MODEL, temperature=1.0)

    invalid = [s for s in STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    log.info("Few-shot examples per prompt: %d", FEW_SHOT_N)

    for strategy in STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)

        few_shot_df = sample_few_shot(full_df, strategy, FEW_SHOT_N)
        few_shot_indices = set(few_shot_df.index)

        if strategy == "D":
            if "relevance" in full_df.columns:
                few_shot = list(zip(
                    few_shot_df["arg1"], few_shot_df["arg2"],
                    few_shot_df["support"], few_shot_df["relevance"],
                ))
            else:
                print("no few shots possible, due to missing relevance score")
                few_shot = []
        else:
            few_shot = list(zip(few_shot_df["arg1"], few_shot_df["arg2"], few_shot_df["support"]))

        eval_df = full_df[~full_df.index.isin(few_shot_indices)]
        if LIMIT is not None:
            eval_df = eval_df.iloc[:LIMIT].copy()

        run_evaluation(
            eval_df, strategy, client,
            BATCH_SIZE, DELAY_SECONDS, MAX_RETRIES, OUTPUT_CSV,
            few_shot=few_shot if few_shot else None,
        )

        if TRACK_RESULTS:
            append_to_overview(
                OUTPUT_CSV, strategy,
                config={
                    "model":          MODEL,
                    "few_shot_n":     len(few_shot),
                    "limit":          LIMIT,
                    "batch_size":     BATCH_SIZE,
                    "delay_seconds":  DELAY_SECONDS,
                    "max_retries":    MAX_RETRIES,
                    "dataset_file":   ARROW_FILE,
                },
            )
            log.info("Overview row appended to ResultOverview_allTests.csv")

    print_all_results(OUTPUT_CSV)
    log.info("Done.")


if __name__ == "__main__":
    main()
