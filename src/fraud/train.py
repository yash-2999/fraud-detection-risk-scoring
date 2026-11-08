import json, logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from .config import ROOT, load_config
from .data import load_dataset, FEATURES
from .split import chronological_split, split_summary
from .preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
def _models(cfg):
    m = cfg["models"]
    return {
      "Logistic Regression": LogisticRegression(max_iter=m["logistic_regression"]["max_iter"], class_weight=m["logistic_regression"]["class_weight"], C=m["logistic_regression"]["C"], random_state=cfg["random_state"]),
      "Decision Tree": DecisionTreeClassifier(max_depth=m["decision_tree"]["max_depth"], min_samples_split=m["decision_tree"]["min_samples_split"], min_samples_leaf=m["decision_tree"]["min_samples_leaf"], class_weight=m["decision_tree"]["class_weight"], random_state=cfg["random_state"]),
      "Random Forest": RandomForestClassifier(n_estimators=m["random_forest"]["n_estimators"], max_depth=m["random_forest"]["max_depth"], min_samples_split=m["random_forest"]["min_samples_split"], min_samples_leaf=m["random_forest"]["min_samples_leaf"], class_weight=m["random_forest"]["class_weight"], random_state=cfg["random_state"], n_jobs=-1)
    }

def train():
    cfg=load_config(); data=load_dataset(ROOT/"data/raw/creditcard.csv")
    splits=chronological_split(data, cfg["split"]["train"], cfg["split"]["validation"])
    (ROOT/"reports").mkdir(exist_ok=True)
    split_summary(splits).to_csv(ROOT/"reports/split_summary.csv", index=False)
    Xtr,ytr=splits["train"][FEATURES],splits["train"]["Class"]
    Xv,yv=splits["validation"][FEATURES],splits["validation"]["Class"]
    results=[]; fitted={}
    for name, estimator in _models(cfg).items():
        pipe=Pipeline([("preprocessor",build_preprocessor()),("model",estimator)])
        pipe.fit(Xtr,ytr)
        prob=pipe.predict_proba(Xv)[:,1]
        row={"model":name,"validation_pr_auc":average_precision_score(yv,prob),"validation_roc_auc":roc_auc_score(yv,prob),"validation_precision":precision_score(yv,prob>=.5,zero_division=0),"validation_recall":recall_score(yv,prob>=.5,zero_division=0),"validation_f1":f1_score(yv,prob>=.5,zero_division=0)}
        results.append(row); fitted[name]=pipe
        logging.info("%s validation PR-AUC %.6f",name,row["validation_pr_auc"])
    comparison=pd.DataFrame(results).sort_values("validation_pr_auc",ascending=False)
    best_name=comparison.iloc[0]["model"]; best=fitted[best_name]
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(best,ROOT/"models/best_model.joblib")
    comparison.to_csv(ROOT/"reports/model_comparison.csv",index=False)
    metadata={"selected_model":best_name,"threshold":cfg["threshold"],"feature_columns":FEATURES,"training_rows":len(splits["train"]),"validation_rows":len(splits["validation"]),"test_rows":len(splits["test"])}
    (ROOT/"models/model_metadata.json").write_text(json.dumps(metadata,indent=2))
    logging.info("Selected %s using validation PR-AUC",best_name)
    return best_name, comparison
if __name__=="__main__": train()
