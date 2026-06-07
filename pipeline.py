"""
Pipeline: dataset loading, evaluation loop, and result saving.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc
from langchain_ollama import ChatOllama

from api_client import call_api, parse_json_array
from evaluation import SUPPORT_MAP
from prompts import STRATEGIES

log = logging.getLogger(__name__)


def load_arrow_dataset(arrow_file: str, limit=None) -> pd.DataFrame:
    arrow_path = Path(arrow_file)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with ipc.open_stream(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()
    df["orig_idx"] = df.index

    # log.info("=== Input file structure ===")
    # log.info("Rows:    %d", len(df))
    # log.info("Columns: %s", list(df.columns))
    # log.info("Dtypes:\n%s", df.dtypes.to_string())
    # log.info("Value counts for 'support' (0=Attack, 1=Support, 2=No Relation):\n%s", df["support"].value_counts().to_string())
    # log.info("Sample row:\n%s", df.iloc[0].to_string())
    # log.info("============================")

    df = df.sample(frac=1).reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit]

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)
    return df


def run_evaluation(
    df: pd.DataFrame,
    strategy: str,
    client: ChatOllama,
    batch_size: int,
    delay_seconds: float,
    max_retries: int,
    output_csv: str,
) -> None:
    """Split the dataset into batches, call the API once per batch, and append results to CSV immediately."""
    prompt_fn = STRATEGIES[strategy]
    total = len(df)
    rows = list(df.itertuples(index=False))

    log.info("Strategy %s  |  %d pairs  |  batch size %d", strategy, total, batch_size)

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]
        batch_end = batch_start + len(batch_rows)
        pairs = [(r.arg1, r.arg2) for r in batch_rows]

        prompt = prompt_fn(pairs)

        # log.info("BATCH %d–%d  (strategy %s)", batch_start + 1, batch_end, strategy)
        # log.info("[PROMPT]\n%s", prompt)

        parsed_batch = None

        try:
            raw = call_api(client, prompt, max_retries)
            # log.info("[LLM RESPONSE]\n%s", raw)
            parsed_batch = parse_json_array(raw, len(batch_rows))
            # log.info("[PARSED]  %s", parsed_batch)
        except Exception as exc:
            log.warning("[FAILED]  Batch %d–%d strategy %s: %s", batch_start + 1, batch_end, strategy, exc)

        batch_results = []
        for i, row in enumerate(batch_rows):
            labels = {"pred_attack": None, "pred_support": None, "pred_neither": None}
            if parsed_batch is not None:
                item = parsed_batch[i]
                for key in ("attack", "support", "neither"):
                    if key in item:
                        labels[f"pred_{key}"] = int(float(item[key]))

            batch_results.append({
                "orig_idx": row.orig_idx,
                "arg1": row.arg1,
                "arg2": row.arg2,
                "support": row.support,
                "strategy": strategy,
                "pred_attack": labels["pred_attack"],
                "pred_support": labels["pred_support"],
                "pred_neither": labels["pred_neither"],
            })

        batch_df = pd.DataFrame(batch_results)
        batch_df["support"] = batch_df["support"].astype(int)
        for col in ("pred_attack", "pred_support", "pred_neither"):
            batch_df[col] = batch_df[col].astype("Int64")
        batch_df.insert(
            batch_df.columns.get_loc("support") + 1,
            "support_label",
            batch_df["support"].map(SUPPORT_MAP),
        )
        batch_df = batch_df.rename(columns={"support": "support [true value]"})

        write_header = not Path(output_csv).exists()
        batch_df.to_csv(output_csv, mode="a", header=write_header, index=False)

        log.info("[PROGRESS]  strategy %s: %d/%d pairs done", strategy, batch_end, total)
        time.sleep(delay_seconds)
