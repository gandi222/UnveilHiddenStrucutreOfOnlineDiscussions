import logging
from pathlib import Path

from langchain_ollama import ChatOllama

from pipeline import load_arrow_dataset, run_evaluation, sample_few_shot
from prompts import STRATEGIES

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "results2.csv"
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
MODEL = "gemma4:latest"
LIMIT = 40          # set to None to use all 4000 pairs
BATCH_SIZE = 10
DELAY_SECONDS = 0
MAX_RETRIES = 1

# "A" — binary, "B" — two-class, "C" — three-class, "D" — three-class + relevance score (zero-shot if dataset has no relevance column), or all at once
STRATEGIES_TO_RUN = ["D"]

# Number of labeled examples injected into each prompt as few-shot context.
# 0 = no few-shot (zero-shot). Sampled once per strategy, balanced across classes.
# For strategy B only Attack/Support examples are eligible (not No Relation).
FEW_SHOT_N = 5
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
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

    log.info("Done. Run evaluation.py for full metrics.")


if __name__ == "__main__":
    main()
