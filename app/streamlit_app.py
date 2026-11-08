import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from src.fraud.config import ROOT as PROJECT_ROOT, load_config
from src.fraud.data import load_dataset, FEATURES
from src.fraud.predict import predict_frame, load_model
from src.fraud.database import store_predictions, query
from src.fraud.sql_analytics import run_analytics
from src.fraud.risk_scoring import risk_category, recommended_action

st.set_page_config(page_title="Fraud Intelligence | Risk Platform",page_icon="◈",layout="wide")
st.markdown('''<style>
.stApp{background:#0b1220;color:#e7eef9}.block-container{padding-top:1.5rem;max-width:1500px}
[data-testid="stMetric"]{background:#111d31;border:1px solid #243752;padding:16px;border-radius:12px}
section[data-testid="stSidebar"]{background:#0e192b}
h1,h2,h3{color:#eaf3ff}.stButton>button{background:#1677d2;color:white;border:0;border-radius:8px}
div[data-testid="stDataFrame"]{border:1px solid #243752;border-radius:10px}
small,.stCaption{color:#9db0c8}
</style>''',unsafe_allow_html=True)
st.sidebar.markdown("## ◈ FRAUD INTELLIGENCE")
st.sidebar.caption("Credit risk analytics • Academic demonstration")
page=st.sidebar.radio("NAVIGATION",["Executive Overview","Data Explorer","Model Performance","Transaction Scoring","Batch Analysis","Risk Analytics","Threshold Simulator","SQL Analytics"])
st.sidebar.divider()
st.sidebar.caption("Model outputs are estimates, not banking decisions.")
cfg=load_config()
def load_artifacts():
    return json.loads((PROJECT_ROOT/"reports/metrics.json").read_text()), pd.read_csv(PROJECT_ROOT/"reports/model_comparison.csv"), load_dataset(PROJECT_ROOT/"data/raw/creditcard.csv")
def safe_artifacts():
    try:return load_artifacts()
    except Exception as e: st.error(str(e)); st.stop()
st.title("Fraud Detection & Risk Scoring")
st.caption("Explainable academic prototype | Chronological evaluation | Imbalanced classification")
if page=="Executive Overview":
    metrics,comparison,df=safe_artifacts()
    st.subheader("Executive overview")
    cols=st.columns(5)
    vals=[("Transactions",f"{len(df):,}"),("Fraud cases",f"{int(df.Class.sum()):,}"),("Legitimate",f"{int((df.Class==0).sum()):,}"),("Fraud prevalence",f"{df.Class.mean()*100:.3f}%"),("Test PR-AUC",f"{metrics['pr_auc']:.4f}")]
    for c,(label,value) in zip(cols,vals): c.metric(label,value)
    c1,c2=st.columns(2)
    c1.plotly_chart(px.pie(df,names="Class",title="Class distribution",hole=.55),use_container_width=True)
    c2.plotly_chart(px.histogram(df,x="Amount",color="Class",nbins=60,title="Transaction amount distribution"),use_container_width=True)
    st.markdown("#### Model status")
    st.write(f"Selected model: **{json.loads((PROJECT_ROOT/'models/model_metadata.json').read_text())['selected_model']}**")
    st.write(f"Classification threshold: **{metrics['threshold']:.2f}**")
    st.info("Dataset is historical and heavily imbalanced. This is not a production banking control.")
    st.plotly_chart(px.bar(comparison,x="model",y="validation_pr_auc",title="Validation PR-AUC (model selection metric)"),use_container_width=True)
elif page=="Data Explorer":
    df=load_dataset(PROJECT_ROOT/"data/raw/creditcard.csv")
    st.subheader("Dataset explorer")
    c=st.columns(4)
    for x,(l,v) in zip(c,[("Rows",f"{len(df):,}"),("Columns",str(df.shape[1])),("Duplicates",f"{df.duplicated().sum():,}"),("Missing",f"{df.isna().sum().sum():,}")]): x.metric(l,v)
    st.dataframe(df.head(100),use_container_width=True)
    st.write("Class distribution"); st.dataframe(df.Class.value_counts().rename_axis("class").reset_index(name="count"))
    st.plotly_chart(px.imshow(df.corr(numeric_only=True),aspect="auto",title="Feature correlation matrix"),use_container_width=True)
    st.plotly_chart(px.histogram(df,x="Time",color="Class",title="Transactions over elapsed time"),use_container_width=True)
elif page=="Model Performance":
    metrics,comparison,df=safe_artifacts()
    st.subheader("Model performance")
    st.markdown("**Validation model selection** — only validation PR-AUC is used to select the model.")
    st.dataframe(comparison,use_container_width=True)
    st.markdown("**Held-out test evaluation**")
    cols=st.columns(5)
    for c,(k,label) in zip(cols,[("pr_auc","PR-AUC"),("roc_auc","ROC-AUC"),("precision","Precision"),("recall","Recall"),("f1","F1")]): c.metric(label,f"{metrics[k]:.4f}")
    st.write("Confusion matrix"); st.dataframe(pd.DataFrame([[metrics["tn"],metrics["fp"]],[metrics["fn"],metrics["tp"]]],index=["Actual legitimate","Actual fraud"],columns=["Predicted legitimate","Predicted fraud"]))
    st.write({k:metrics[k] for k in ["tn","fp","fn","tp","specificity","false_positive_rate","random_baseline_pr_auc"]})
    for fn in ["roc_curve.png","precision_recall_curve.png","confusion_matrix.png"]:
        p=PROJECT_ROOT/"reports/figures"/fn
        if p.exists(): st.image(str(p),caption=fn.replace("_"," ").replace(".png","").title(),width=550)
elif page=="Transaction Scoring":
    st.subheader("Individual transaction scoring")
    st.warning("Manually entered feature vectors are demonstration inputs; arbitrary V-feature values may be outside the training distribution.")
    with st.form("single"):
        amount=st.number_input("Transaction amount",min_value=0.0,value=100.0)
        t=st.number_input("Time (seconds from dataset reference)",min_value=0.0,value=10000.0)
        values={}
        for i in range(1,29): values[f"V{i}"]=st.number_input(f"V{i}",value=0.0,format="%.5f")
        submitted=st.form_submit_button("Score transaction")
    if submitted:
        row={"Time":t,"Amount":amount,**values}
        result=predict_frame(pd.DataFrame([row]))
        r=result.iloc[0]; st.metric("Estimated fraud probability",f"{r.fraud_probability*100:.3f}%")
        st.subheader(f"Risk category: {r.risk_category}")
        st.write(recommended_action(r.risk_category))
        st.caption("Model probability is not necessarily a calibrated real-world probability.")
elif page=="Batch Analysis":
    st.subheader("Batch CSV analysis")
    st.write("CSV must contain Time, Amount, and V1 through V28. Class is optional and ignored for inference.")
    upload=st.file_uploader("Upload transaction CSV",type=["csv"])
    threshold=st.slider("Classification threshold",0.01,0.99,float(cfg["threshold"]),0.01)
    if upload:
        try:
            batch=pd.read_csv(upload)
            st.dataframe(batch.head(10),use_container_width=True)
            if st.button("Run batch predictions"):
                result=predict_frame(batch,threshold=threshold)
                st.session_state["batch_result"]=result
                stored=store_predictions(result)
                st.success(f"Scored {len(result):,} rows; stored {stored:,} prediction records in SQLite.")
        except Exception as e: st.error(f"CSV rejected: {e}")
    if "batch_result" in st.session_state:
        result=st.session_state["batch_result"]
        c=st.columns(4)
        for x,(l,v) in zip(c,[("Processed",len(result)),("Predicted fraud",int(result.prediction.sum())),("Average score",f"{result.fraud_probability.mean():.3%}"),("High/critical",int(result.risk_category.isin(["HIGH","CRITICAL"]).sum()))]): x.metric(l,v)
        st.dataframe(result,use_container_width=True)
        st.download_button("Download scored CSV",result.to_csv(index=False),"fraud_predictions.csv","text/csv")
elif page=="Risk Analytics":
    st.subheader("Risk analytics")
    metrics,comparison,df=safe_artifacts()
    test=pd.read_csv(PROJECT_ROOT/"reports/test_predictions.csv")
    test["risk_category"]=test.probability.map(risk_category)
    st.plotly_chart(px.histogram(test,x="probability",color="risk_category",nbins=50,title="Predicted risk score distribution"),use_container_width=True)
    st.dataframe(test.groupby("risk_category").agg(transactions=("prediction","size"),predicted_fraud=("prediction","sum"),mean_score=("probability","mean")).reset_index())
    st.caption("Risk bands are illustrative configuration values, not regulatory or bank policy.")
elif page=="Threshold Simulator":
    st.subheader("Threshold simulator")
    metrics,comparison,df=safe_artifacts()
    test=pd.read_csv(PROJECT_ROOT/"reports/test_predictions.csv")
    threshold=st.slider("Threshold",0.01,0.99,0.50,0.01)
    from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
    pred=(test.probability>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(test.y_true,pred,labels=[0,1]).ravel()
    c=st.columns(5)
    for x,(l,v) in zip(c,[("Precision",precision_score(test.y_true,pred,zero_division=0)),("Recall",recall_score(test.y_true,pred,zero_division=0)),("F1",f1_score(test.y_true,pred,zero_division=0)),("False positives",fp),("False negatives",fn)]): x.metric(l,f"{v:.4f}" if isinstance(v,float) else f"{v:,}")
    st.dataframe(pd.DataFrame([[tn,fp],[fn,tp]],index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]))
    st.warning("This view is retrospective test-set sensitivity analysis only. Do not choose a deployed threshold from test labels; use validation data.")
elif page=="SQL Analytics":
    st.subheader("SQL analytics")
    tabs=run_analytics()
    for name,rows in tabs.items():
        st.markdown(f"**{name}**")
        st.dataframe(pd.DataFrame(rows))
    st.caption("Analytics reflect predictions persisted from batch scoring sessions in this local SQLite database.")
