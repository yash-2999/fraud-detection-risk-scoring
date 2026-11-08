import numpy as np
import pandas as pd
import pytest
from src.fraud.data import validate_dataset, FEATURES
from src.fraud.split import chronological_split
from src.fraud.risk_scoring import risk_category
from src.fraud.predict import predict_frame
from src.fraud.database import store_predictions, query

def test_validate(tiny_df):
    result=validate_dataset(tiny_df); assert result["rows"]==100 and result["fraud_count"]==10
def test_validation_rejects_missing(tiny_df):
    with pytest.raises(ValueError): validate_dataset(tiny_df.drop(columns=["V1"]))
def test_chronological_split(tiny_df):
    s=chronological_split(tiny_df); assert len(s["train"])==60 and len(s["validation"])==20 and len(s["test"])==20
    assert s["train"].Time.max() < s["validation"].Time.min() < s["test"].Time.min()
def test_risk_categories():
    assert risk_category(.02)=="LOW" and risk_category(.2)=="MEDIUM" and risk_category(.5)=="HIGH" and risk_category(.8)=="CRITICAL"
def test_invalid_probability():
    with pytest.raises(ValueError): risk_category(1.2)
def test_prediction_and_db(tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    X=pd.DataFrame([{c:0.0 for c in FEATURES},{c:1.0 for c in FEATURES},{c:2.0 for c in FEATURES},{c:3.0 for c in FEATURES}])
    y=[0,0,1,1]
    pre=ColumnTransformer([("n",Pipeline([("imp",SimpleImputer()),("sc",StandardScaler())]),FEATURES)])
    model=Pipeline([("preprocessor",pre),("model",LogisticRegression())]).fit(X,y)
    out=predict_frame(X,model=model)
    assert out.fraud_probability.between(0,1).all()
    db=tmp_path/"test.db"; assert store_predictions(out,db_path=db)==4
    assert query("SELECT COUNT(*) AS n FROM predictions",db_path=db)[0]["n"]==4

def test_all_sql_analytics_execute(tmp_path, monkeypatch):
    from src.fraud import database
    from src.fraud.sql_analytics import ANALYTICS
    db=tmp_path/"analytics.db"
    for name,sql in ANALYTICS.items():
        rows=database.query(sql,db_path=db)
        assert isinstance(rows,list), name
