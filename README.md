# Fraud Detection & Risk Scoring System

A modular academic ML application for credit-card transaction risk estimation using the Kaggle ULB dataset, scikit-learn pipelines, Streamlit, Plotly and SQLite. It is a portfolio demonstration, not a production banking fraud-control system.

## Dataset
Place the original `creditcard.csv` from [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) in `data/raw/creditcard.csv`. Expected source dataset: 284,807 rows, 492 frauds, 30 features plus `Class`. Dataset is excluded from Git by `.gitignore`.

## Methodology
- Sort by `Time`, then split first 60% train / next 20% validation / final 20% test.
- Fit imputing/scaling transformations within each model pipeline, on training rows only.
- Compare Logistic Regression, Decision Tree and Random Forest by validation PR-AUC.
- Evaluate the validation-selected model once on the held-out chronological test partition.
- Default classification threshold is 0.50; probability estimates may not be calibrated.
- Random baseline PR-AUC is test fraud prevalence.

## Run
See [RUN_GUIDE.md](RUN_GUIDE.md) for Windows PowerShell instructions.

## Pages
Executive Overview, Data Explorer, Model Performance, Transaction Scoring, Batch Analysis, Risk Analytics, Threshold Simulator and SQL Analytics.

## Structure
See the repository tree. Main ML modules are under `src/fraud/`; Streamlit UI is `app/streamlit_app.py`.

## Limitations
The source dataset is historical and anonymized. Time-based split tests temporal generalization within this dataset but does not represent a live deployment distribution. The sample has no cardholder identity, merchant, geography or real-time context. No probability calibration or cost-sensitive operational policy is claimed. Manual V1–V28 inputs are not meaningful unless derived from the same transformation process as the original dataset.

## Screenshots
Capture screenshots locally after launching the app; the project does not claim screenshots of an unlaunched UI.


## Reproduced run (included model/report)
Selected model: **Logistic Regression**. Validation PR-AUCs: Logistic Regression=0.771877, Random Forest=0.747472, Decision Tree=0.590523. Held-out test PR-AUC=0.743626, ROC-AUC=0.982026, precision=4.5362%, recall=89.3333%. Confusion matrix TN=55477, FP=1410, FN=8, TP=67. See `reports/evaluation_report.md` and `reports/metrics.json`.
