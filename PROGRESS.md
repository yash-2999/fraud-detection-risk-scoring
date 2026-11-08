# Project Progress

## Implemented
- [x] Modular source layout, YAML configuration, logging and type-friendly functions
- [x] Real Kaggle CSV copied into working project (excluded from ZIP to avoid a 150 MB dataset bundle)
- [x] Dataset schema and numerical validation
- [x] Chronological 60/20/20 split with split summary
- [x] Logistic Regression, Decision Tree and Random Forest training pipelines
- [x] Validation PR-AUC selection and persisted best pipeline
- [x] Held-out test evaluation, metrics JSON, classification report and plots
- [x] Probability prediction and configurable risk categories
- [x] SQLite transaction, prediction and model metadata schema
- [x] SQL analytics and parameterized query helper
- [x] Eight-page Streamlit dashboard source
- [x] Batch CSV prediction and download source
- [x] Pytest suite, notebook and documentation

## Executed verification
- [x] Actual dataset training completed
- [x] Held-out evaluation completed
- [x] Python source compilation completed
- [x] Pytest: 6 passed
- [x] SQLite insert/query exercised by test

## Not verified in this hosted build environment
- [ ] Streamlit browser launch and page-by-page visual QA: Streamlit is not installed in this runtime; install from requirements.txt on Windows.
- [ ] Browser screenshots: not captured because no browser UI runtime is available.
- [ ] End-to-end UI testing of file upload/download.

## Known limitations
- The real 150 MB CSV is not bundled in the ZIP; retrieve it from Kaggle and place it in data/raw/.
- Probability calibration, deployment monitoring and production controls are out of scope.
