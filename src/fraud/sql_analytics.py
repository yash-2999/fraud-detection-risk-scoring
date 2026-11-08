from .database import query

ANALYTICS={
 "Transaction totals":"SELECT COUNT(*) AS total_transactions FROM transactions",
 "Prediction totals":"SELECT COUNT(*) AS total_predictions, SUM(prediction) AS predicted_fraud FROM predictions",
 "Risk distribution":"SELECT risk_category, COUNT(*) AS transactions FROM predictions GROUP BY risk_category ORDER BY transactions DESC",
 "Amount by risk":"SELECT risk_category, COUNT(*) AS transactions, ROUND(AVG(amount),2) AS average_amount, ROUND(SUM(amount),2) AS total_amount FROM predictions GROUP BY risk_category",
 "Daily summary":"SELECT CAST(transaction_time / 86400 AS INTEGER) AS day_index, COUNT(*) AS transactions, SUM(prediction) AS predicted_fraud FROM predictions GROUP BY day_index ORDER BY day_index",
 "Model registry":"SELECT model_name, model_version, registered_at FROM model_metadata ORDER BY registered_at DESC"
}
def run_analytics(): return {name:query(sql) for name,sql in ANALYTICS.items()}
