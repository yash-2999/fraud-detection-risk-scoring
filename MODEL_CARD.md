# Model Card

## Intended use
Educational demonstration of imbalanced binary classification and transaction risk scoring.

## Data
Kaggle ULB European cardholder transactions. Features are anonymized PCA components V1–V28 plus Time and Amount; target `Class`.

## Training
Chronological 60/20/20 split. Preprocessing is fit inside sklearn pipelines on training data only. Model selection uses validation PR-AUC among Logistic Regression, Decision Tree and Random Forest.

## Evaluation
Run `python -m src.fraud.evaluate` to generate actual held-out metrics in `reports/metrics.json`. The random PR-AUC baseline is the observed fraud prevalence in the test partition.

## Limitations and risks
Historical anonymized data; temporal drift; no merchant/cardholder context; class imbalance; possible probability miscalibration; no formal fairness, privacy, adversarial robustness or production latency assessment. Do not use to decline transactions or make financial decisions.
