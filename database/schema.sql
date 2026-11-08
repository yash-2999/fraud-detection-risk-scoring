CREATE TABLE IF NOT EXISTS transactions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 transaction_time REAL NOT NULL,
 amount REAL NOT NULL,
 ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 transaction_id INTEGER,
 transaction_time REAL NOT NULL,
 amount REAL NOT NULL,
 fraud_probability REAL NOT NULL CHECK(fraud_probability >= 0 AND fraud_probability <= 1),
 risk_category TEXT NOT NULL CHECK(risk_category IN ('LOW','MEDIUM','HIGH','CRITICAL')),
 prediction INTEGER NOT NULL CHECK(prediction IN (0,1)),
 model_name TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(transaction_id) REFERENCES transactions(id)
);
CREATE TABLE IF NOT EXISTS model_metadata (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 model_name TEXT NOT NULL,
 model_version TEXT NOT NULL,
 metadata_json TEXT NOT NULL,
 registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_category);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
