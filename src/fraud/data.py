from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)] + ["Class"]
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Download the Kaggle ULB dataset and place it at data/raw/creditcard.csv.")
    df = pd.read_csv(path)
    validate_dataset(df)
    return df

def validate_dataset(df, require_target=True):
    required = EXPECTED if require_target else FEATURES
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_target and not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 and 1.")
    numeric = required
    non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")
    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if np.isinf(df[required].to_numpy(dtype=float)).any():
        raise ValueError("Required columns contain infinite values.")
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_count": int(df["Class"].sum()) if require_target else None,
        "fraud_rate": float(df["Class"].mean()) if require_target else None,
    }
