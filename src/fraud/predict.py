import joblib
import pandas as pd
from .config import ROOT
from .data import FEATURES, validate_dataset
from .risk_scoring import risk_category, recommended_action

def load_model():
    path=ROOT/"models/best_model.joblib"
    if not path.exists(): raise FileNotFoundError("Model not found. Run: python -m src.fraud.train")
    return joblib.load(path)

def predict_frame(frame, threshold=0.5, model=None):
    missing=sorted(set(FEATURES)-set(frame.columns))
    if missing: raise ValueError(f"Missing feature columns: {missing}")
    X=frame[FEATURES].copy()
    if X.empty: raise ValueError("Input CSV contains no transactions.")
    if X.isna().any().any(): raise ValueError("Input features contain missing values.")
    import numpy as np
    if not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError("Input features contain infinite or non-finite values.")
    model=model or load_model()
    prob=model.predict_proba(X)[:,1]
    out=frame.copy(); out["fraud_probability"]=prob; out["risk_category"]=[risk_category(float(x)) for x in prob]; out["prediction"]=(prob>=threshold).astype(int); out["recommended_action"]=out["risk_category"].map(recommended_action)
    return out
