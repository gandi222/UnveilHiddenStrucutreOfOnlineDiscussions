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


def load_dataset(path: str, limit=None) -> pd.DataFrame:
    """Unified loader: dispatches to CSV or Arrow loader based on file extension."""
    if path.endswith(".csv"):
        return load_csv_dataset(path, limit)
    return load_arrow_dataset(path, limit)


def load_csv_dataset(csv_file: str, limit=None) -> pd.DataFrame:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"arg1", "arg2", "support"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.sample(frac=1).reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit]

    if "orig_idx" not in df.columns:
        df["orig_idx"] = df.index

    log.info("Loaded %d argument pairs from %s", len(df), csv_path)
    return df


def load_arrow_dataset(arrow_file: str, limit=None) -> pd.DataFrame:
    arrow_path = Path(arrow_file)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with ipc.open_stream(str(arrow_path)) as reader:
        table = reader.read_all()

    df = table.to_pandas()
    df["orig_idx"] = df.index

    df = df.sample(frac=1).reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit]

    log.info("Loaded %d argument pairs from %s", len(df), arrow_path)
    return df


def sample_few_shot(df: pd.DataFrame, strategy: str, n: int) -> pd.DataFrame:
    """Return a DataFrame of n balanced few-shot rows sampled from df.

    Strategy B can only demonstrate Attack and Support (not No Relation), so
    No Relation rows are excluded from its pool.
    Sampling is balanced: roughly equal representation per eligible class.
    The returned DataFrame preserves df's index so callers can exclude these
    rows from the evaluation set.
    """
    if n == 0:
        return df.iloc[:0]  # empty DataFrame with correct columns

    eligible_classes = [0, 1, 2] if strategy in ("C", "D") else [0, 1]
    pool = df[df["support"].isin(eligible_classes)]

    k = len(eligible_classes)
    per_class, remainder = divmod(n, k)

    selected = []
    for i, cls in enumerate(eligible_classes):
        take = per_class + (1 if i < remainder else 0)
        cls_rows = pool[pool["support"] == cls]
        take = min(take, len(cls_rows))
        if take > 0:
            selected.append(cls_rows.sample(take))

    return pd.concat(selected).sample(frac=1)


def run_evaluation(
    df: pd.DataFrame,
    strategy: str,
    client: ChatOllama,
    batch_size: int,
    delay_seconds: float,
    max_retries: int,
    output_csv: str,
    few_shot: list = None,
) -> None:
    """Split the dataset into batches, call the API once per batch, and append results to CSV immediately.

    few_shot: pre-sampled list of (arg1, arg2, support_int) tuples, or None for zero-shot.
    The caller is responsible for ensuring these rows are excluded from df.
    """
    prompt_fn = STRATEGIES[strategy]
    total = len(df)
    rows = list(df.itertuples(index=False))

    if few_shot is None:
        few_shot = []

    log.info(
        "Strategy %s  |  %d pairs  |  batch size %d  |  few-shot examples: %d",
        strategy, total, batch_size, len(few_shot),
    )
    if few_shot:
        for i, (arg1, arg2, support, *_rest) in enumerate(few_shot, start=1):
            log.info(
                "  Few-shot example %d: label=%s (%d) | Arg1: %.60s | Arg2: %.60s",
                i, SUPPORT_MAP[support], support, arg1, arg2,
            )

    _printed_first_prompt = False
    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]
        batch_end = batch_start + len(batch_rows)
        pairs = [(r.arg1, r.arg2) for r in batch_rows]

        prompt = prompt_fn(pairs, few_shot=few_shot if few_shot else None)

        if not _printed_first_prompt:
            print(f"\n{'='*60}\n[FULL PROMPT — strategy {strategy}, batch 1]\n{'='*60}\n{prompt}\n{'='*60}\n")
            _printed_first_prompt = True
        # log.info("BATCH %d–%d  (strategy %s)", batch_start + 1, batch_end, strategy)

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
            labels = {"pred_attack": None, "pred_support": None, "pred_neither": None, "pred_relevance": None}
            if parsed_batch is not None:
                item = parsed_batch[i]
                for key in ("attack", "support", "neither"):
                    if key in item:
                        labels[f"pred_{key}"] = int(float(item[key]))
                if strategy == "D" and "relevance" in item:
                    labels["pred_relevance"] = float(item["relevance"])

            batch_results.append({
                "orig_idx": row.orig_idx,
                "arg1": row.arg1,
                "arg2": row.arg2,
                "support": row.support,
                "strategy": strategy,
                "pred_attack": labels["pred_attack"],
                "pred_support": labels["pred_support"],
                "pred_neither": labels["pred_neither"],
                "pred_relevance": labels["pred_relevance"],
                "relevance human labeled": getattr(row, "relevance_human_labeled", None),
            })

        batch_df = pd.DataFrame(batch_results)
        batch_df["support"] = batch_df["support"].astype(int)
        for col in ("pred_attack", "pred_support", "pred_neither"):
            batch_df[col] = batch_df[col].astype("Int64")
        batch_df["pred_relevance"] = batch_df["pred_relevance"].astype("Float64")
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
