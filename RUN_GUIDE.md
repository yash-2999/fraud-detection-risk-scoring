# Windows 10/11 + VS Code run guide

## 1. Extract and open
Extract `fraud-detection-risk-scoring.zip`, then open the extracted folder in VS Code.

## 2. Create and activate environment (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked for this PowerShell session only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Dataset
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at `data/raw/creditcard.csv`. Do not rename columns.

## 4. Train and evaluate
```powershell
python -m src.fraud.train
python -m src.fraud.evaluate
```
Training saves the validation-selected pipeline to `models/best_model.joblib`; evaluation writes metrics and plots to `reports/`.

## 5. Tests
```powershell
python -m pytest -q
```

## 6. Launch dashboard
```powershell
streamlit run app/streamlit_app.py
```
Open http://localhost:8501. Stop with Ctrl+C. Reopen by rerunning Streamlit; do not retrain unless the dataset/config/model changes.

## 7. Batch prediction
Open Batch Analysis, upload a CSV containing `Time`, `Amount`, and `V1` through `V28`, click Run batch predictions, then download the scored CSV.

## Troubleshooting
- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- Dataset not found: verify exact path `data/raw/creditcard.csv`.
- Missing feature columns: check all 30 feature names and capitalization.
- Port in use: run `streamlit run app/streamlit_app.py --server.port 8502`.
- If `py -3.11` is unavailable, install Python 3.11 and enable the Python Launcher.
