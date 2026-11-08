import json
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve)
from .config import ROOT, load_config
from .data import load_dataset, FEATURES
from .split import chronological_split

def evaluate():
    cfg=load_config(); df=load_dataset(ROOT/"data/raw/creditcard.csv")
    splits=chronological_split(df,cfg["split"]["train"],cfg["split"]["validation"])
    test=splits["test"]; model=joblib.load(ROOT/"models/best_model.joblib")
    X,y=test[FEATURES],test["Class"]; p=model.predict_proba(X)[:,1]; threshold=float(cfg["threshold"]); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    metrics={"threshold":threshold,"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),"random_baseline_pr_auc":float(y.mean()),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"specificity":float(tn/(tn+fp)) if tn+fp else 0.0,"false_positive_rate":float(fp/(fp+tn)) if fp+tn else 0.0,"test_rows":len(test),"test_fraud":int(y.sum())}
    (ROOT/"reports/metrics.json").write_text(json.dumps(metrics,indent=2))
    (ROOT/"reports/classification_report.txt").write_text(classification_report(y,pred,zero_division=0))
    pd.DataFrame({"y_true":y,"probability":p,"prediction":pred}).to_csv(ROOT/"reports/test_predictions.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y,pred,ax=ax,colorbar=False); fig.tight_layout(); fig.savefig(ROOT/"reports/figures/confusion_matrix.png",dpi=150); plt.close(fig)
        fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],"--"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curve"); fig.tight_layout(); fig.savefig(ROOT/"reports/figures/roc_curve.png",dpi=150); plt.close(fig)
        precision,recall,_=precision_recall_curve(y,p); fig,ax=plt.subplots(); ax.plot(recall,precision); ax.axhline(y.mean(),linestyle="--"); ax.set(xlabel="Recall",ylabel="Precision",title="Precision-recall curve"); fig.tight_layout(); fig.savefig(ROOT/"reports/figures/precision_recall_curve.png",dpi=150); plt.close(fig)
    except Exception as exc:
        metrics["plot_generation_warning"]=str(exc)
        (ROOT/"reports/metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics
if __name__=="__main__":
    print(json.dumps(evaluate(),indent=2))
