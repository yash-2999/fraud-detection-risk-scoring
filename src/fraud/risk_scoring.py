from .config import load_config

def risk_category(probability, bands=None):
    if not 0 <= float(probability) <= 1: raise ValueError("Probability must be between 0 and 1.")
    if bands is None:
        bands=load_config()["risk_bands"]
    if probability < bands["low"]: return "LOW"
    if probability < bands["medium"]: return "MEDIUM"
    if probability < bands["high"]: return "HIGH"
    return "CRITICAL"

def recommended_action(category):
    return {"LOW":"Normal monitoring.","MEDIUM":"Review if additional suspicious indicators exist.","HIGH":"Prioritize manual transaction review.","CRITICAL":"Escalate for immediate investigation."}[category]
