import sqlite3
from pathlib import Path
from .config import ROOT

DB_PATH=ROOT/"database/fraud_detection.db"
SCHEMA=ROOT/"database/schema.sql"

def connect(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(db_path); con.row_factory=sqlite3.Row
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    return con

def store_predictions(frame, model_name="unknown", db_path=DB_PATH):
    con=connect(db_path)
    try:
        count=0
        for _,r in frame.iterrows():
            cur=con.execute("INSERT INTO transactions(transaction_time, amount) VALUES(?,?)",
                            (float(r.get("Time",0)),float(r.get("Amount",0))))
            con.execute("INSERT INTO predictions(transaction_id, transaction_time, amount, fraud_probability, risk_category, prediction, model_name) VALUES(?,?,?,?,?,?,?)",
                        (cur.lastrowid,float(r.get("Time",0)),float(r.get("Amount",0)),float(r["fraud_probability"]),str(r["risk_category"]),int(r["prediction"]),model_name))
            count+=1
        con.commit()
        return count
    finally: con.close()

def query(sql, params=(), db_path=DB_PATH):
    con=connect(db_path)
    try: return [dict(r) for r in con.execute(sql,params).fetchall()]
    finally: con.close()

def register_model(model_name, metadata, db_path=DB_PATH):
    import json
    con=connect(db_path)
    try:
        con.execute("INSERT INTO model_metadata(model_name, model_version, metadata_json) VALUES(?,?,?)",
                    (model_name, "1.0.0", json.dumps(metadata)))
        con.commit()
    finally: con.close()
