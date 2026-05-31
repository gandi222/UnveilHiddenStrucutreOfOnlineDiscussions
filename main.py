import logging
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

import config
from evaluation import SUPPORT_MAP, add_correct_column, accuracy_summary
from pipeline import load_arrow_dataset, save_results, run_evaluation
from prompts import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    Path(config.OUTPUT_CSV).unlink(missing_ok=True)
    log.info("Cleared %s (fresh run)", config.OUTPUT_CSV)

    df = load_arrow_dataset(config.ARROW_FILE, limit=config.LIMIT)
    client = ChatOllama(base_url=config.BASE_URL, model=config.MODEL, temperature=1.0)

    invalid = [s for s in config.STRATEGIES_TO_RUN if s not in STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Must be one of {list(STRATEGIES.keys())}")

    all_results = []
    for strategy in config.STRATEGIES_TO_RUN:
        log.info("Running strategy %s", strategy)
        all_results.extend(run_evaluation(df, strategy, client))

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

    save_results(out_df, config.OUTPUT_CSV)

    log.info("=== Accuracy summary ===")
    for strat, acc in accuracy_summary(out_df).items():
        grp = out_df[out_df["strategy"] == strat]["correct"].dropna()
        log.info("Strategy %s: %d/%d correct (%.1f%%)", strat, grp.sum(), len(grp), 100 * acc)
    log.info("========================")


if __name__ == "__main__":
    main()
