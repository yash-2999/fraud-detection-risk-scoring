import pandas as pd
from .data import FEATURES

def chronological_split(df, train_fraction=0.60, validation_fraction=0.20):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    splits = {"train": ordered.iloc[:a].copy(), "validation": ordered.iloc[a:b].copy(), "test": ordered.iloc[b:].copy()}
    if any(len(x) == 0 for x in splits.values()):
        raise ValueError("A split is empty.")
    return splits

def split_summary(splits):
    rows = []
    for name, part in splits.items():
        fraud = int(part["Class"].sum())
        rows.append({"split": name, "rows": len(part), "fraud": fraud, "legitimate": len(part)-fraud, "fraud_prevalence": part["Class"].mean(), "min_time": part.Time.min(), "max_time": part.Time.max()})
    return pd.DataFrame(rows)
