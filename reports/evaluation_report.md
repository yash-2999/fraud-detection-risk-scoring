# Evaluation Report

## Protocol
Chronological sort by `Time`; first 60% training, next 20% validation, final 20% test. Model selection uses validation PR-AUC only. Preprocessing is fitted within each model pipeline on training rows only.

## Split summary
| split      |   rows |   fraud |   legitimate |   fraud_prevalence |   min_time |   max_time |
|:-----------|-------:|--------:|-------------:|-------------------:|-----------:|-----------:|
| train      | 170884 |     360 |       170524 |         0.00210669 |          0 |     120396 |
| validation |  56961 |      57 |        56904 |         0.00100068 |     120396 |     145247 |
| test       |  56962 |      75 |        56887 |         0.00131667 |     145248 |     172792 |

## Validation model comparison
| model               |   validation_pr_auc |   validation_roc_auc |   validation_precision |   validation_recall |   validation_f1 |
|:--------------------|--------------------:|---------------------:|-----------------------:|--------------------:|----------------:|
| Logistic Regression |            0.771877 |             0.970927 |              0.0269699 |            0.894737 |       0.0523614 |
| Random Forest       |            0.747472 |             0.937874 |              0.95122   |            0.684211 |       0.795918  |
| Decision Tree       |            0.590523 |             0.884139 |              0.0507497 |            0.77193  |       0.0952381 |

Selected model: **Logistic Regression**, selected by validation PR-AUC.

## Held-out test metrics at threshold 0.50
| Metric | Value |
|---|---:|
| Accuracy | 0.975106 |
| Precision | 0.045362 |
| Recall | 0.893333 |
| F1 | 0.086340 |
| ROC-AUC | 0.982026 |
| PR-AUC | 0.743626 |
| Random baseline PR-AUC (test fraud prevalence) | 0.001317 |
| Specificity | 0.975214 |
| False-positive rate | 0.024786 |

Confusion matrix: TN=55,477, FP=1,410, FN=8, TP=67. Test rows=56,962; fraud cases=75.

## Interpretation
The model detects 67 of 75 fraud cases at this threshold, with 1,410 false-positive alerts. Accuracy alone is misleading because legitimate transactions dominate. The random baseline PR-AUC equals observed test fraud prevalence. Scores have not been calibrated as real-world probabilities.

The result depends on the dataset, package versions and configuration. It is not a production banking system.
