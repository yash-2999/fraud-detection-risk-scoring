# Interview Guide — Fraud Detection & Risk Scoring

1. **Why is fraud detection imbalanced?** Fraud is a tiny minority class; accuracy can be high while fraud recall is poor.
2. **Why PR-AUC?** It emphasizes precision-recall trade-offs and is informative when positives are rare.
3. **What is ROC-AUC?** Probability that a random positive receives a higher score than a random negative.
4. **Why chronological splitting?** It approximates future-on-past deployment and reduces temporal leakage.
5. **What is leakage?** Information from validation/test influencing training or model selection.
6. **Why fit scalers only on train?** Global fitting leaks distributional information from future partitions.
7. **Why use Pipeline?** It binds preprocessing and estimator so transformations are consistently fitted/applied.
8. **Why Logistic Regression?** Strong interpretable baseline with probabilistic output.
9. **Why Decision Tree?** Captures nonlinear rules but can overfit.
10. **Why Random Forest?** Bagged trees reduce variance; computationally more expensive.
11. **What is class_weight='balanced'?** Weights inversely proportional to class frequency.
12. **Why not blindly use SMOTE?** Synthetic interpolation can distort temporal fraud patterns and must be confined to training.
13. **What does precision mean?** Fraction of flagged transactions that are actually fraud.
14. **What does recall mean?** Fraction of fraud cases detected.
15. **What is F1?** Harmonic mean of precision and recall.
16. **What is PR-AUC baseline?** Approximately positive-class prevalence for a random ranking.
17. **Why can accuracy mislead?** Predicting every row legitimate yields about 99.8% accuracy here.
18. **What is a false positive?** Legitimate transaction incorrectly flagged.
19. **What is a false negative?** Fraud transaction missed by the model.
20. **What is specificity?** True-negative rate among legitimate transactions.
21. **What is threshold tuning?** Converting scores to labels at a chosen operating point.
22. **Why tune on validation?** Test data must remain untouched for final unbiased evaluation.
23. **Are model probabilities calibrated?** Not necessarily; calibration must be separately measured and fitted.
24. **What does risk band mean?** A configurable mapping from score intervals to review priorities.
25. **Why SQLite?** Lightweight embedded persistence for a local demo.
26. **Why parameterized SQL?** Prevents SQL injection and handles values safely.
27. **Why Streamlit?** Rapid Python-native interactive data application framework.
28. **Why persist the entire pipeline?** Prediction requires the same fitted preprocessing and estimator.
29. **What is temporal drift?** Changes in transaction/fraud distributions over time.
30. **What would production require?** Streaming ingestion, online feature engineering, calibrated cost-sensitive policy, monitoring, auditability, security and human escalation.
31. **Why can precision be low?** Rare fraud means even a low false-positive rate can generate many alerts.
32. **What is a confusion matrix?** Counts TP, TN, FP and FN at a fixed threshold.
33. **Why inspect split prevalence?** Temporal segments may have different fraud rates and affect metric stability.
34. **What is a limitation of V1–V28?** They are anonymized components with little direct business interpretability.
35. **How to improve?** Calibrate scores, tune threshold on validation with explicit costs, monitor drift and evaluate on newer external data.
