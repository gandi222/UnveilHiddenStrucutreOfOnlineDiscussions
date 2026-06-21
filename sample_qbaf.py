"""
Sample 100 random argument pairs from NR_WebDataset and save to QBAF_relevanceScore.csv.

The output CSV is intended for manual relevance annotation. Fill in the `relevance`
column (values: 0.00, 0.25, 0.50, 0.75, 1.00) and use it as INPUT_FILE in main.py
to run strategy D with few-shot relevance examples.
"""

import pandas as pd

from evaluation import SUPPORT_MAP
from pipeline import load_arrow_dataset

ARROW_FILE = "NR_WebDataset/data-00000-of-00001.arrow"
OUTPUT_CSV = "QBAF_relevanceScore.csv"
N_SAMPLES = 100

df = load_arrow_dataset(ARROW_FILE)
df = df.iloc[:N_SAMPLES].copy()

df.insert(df.columns.get_loc("support") + 1, "support_label", df["support"].map(SUPPORT_MAP))
df["relevance"] = pd.NA

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
print(df["support_label"].value_counts().to_string())
