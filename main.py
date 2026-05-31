import logging
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

from evaluation import SUPPORT_MAP, add_correct_column, accuracy_summary
from pipeline import load_arrow_dataset, save_results, run_evaluation
from prompts import STRATEGIES

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------
ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "results2.csv"
BASE_URL = "https://ollama-gpt-oss.cluster.ai.wu.ac.at/"
MODEL = "gemma4:latest"
LIMIT = 20          # set to None to use all 1284 pairs
BATCH_SIZE = 10
DELAY_SECONDS = 0.5
MAX_RETRIES = 1

# "A" — binary, "B" — two-class, "C" — three-class
STRATEGIES_TO_RUN = ["A"]
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    Path(OUTPUT_CSV).unlink(missing_ok=True)
    log.info("Cleared %s (fresh run)", OUTPUT_CSV)

    df = load_arrow_dataset(ARROW_FILE, limit=LIMIT)
    client = ChatOllama(base_url=BASE_URL, model=MODEL, temperature=1.0)

    invalid = [s for s in STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    all_results = []
    for strategy in STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)
        all_results.extend(run_evaluation(df, strategy, client, BATCH_SIZE, DELAY_SECONDS, MAX_RETRIES))

    out_df = pd.DataFrame(all_results, columns=[
        "orig_idx", "arg1", "arg2", "support", "strategy",
        "pred_attack", "pred_support", "pred_neither",
    ])

    out_df["support"] = out_df["support"].astype(int)
    for col in ("pred_attack", "pred_support", "pred_neither"):
        out_df[col] = out_df[col].astype("Int64")

    out_df.insert(
        out_df.columns.get_loc("support") + 1,
        "support_label",
        out_df["support"].map(SUPPORT_MAP),
    )

    add_correct_column(out_df)
    out_df = out_df.rename(columns={"support": "support [true value]"})

    save_results(out_df, OUTPUT_CSV)

    log.info("=== Accuracy summary ===")
    for strat, acc in accuracy_summary(out_df).items():
        grp = out_df[out_df["strategy"] == strat]["correct"].dropna()
        log.info("Strategy %s: %d/%d correct (%.1f%%)", strat, grp.sum(), len(grp), 100 * acc)
    log.info("========================")


if __name__ == "__main__":
    main()
