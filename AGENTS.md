# Auto Classifier Benchmarking System (acbs)

A system that takes a tabular dataset and target column, preprocesses it, trains and tunes
classifiers (Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, Gradient Boosting,
Naive Bayes, KNN) with stratified k-fold cross-validation, measures Accuracy, Precision, Recall,
F1, ROC-AUC, training time and prediction time, ranks models using user-defined weights,
recommends the best model with a textual explanation, stores every experiment in a database,
and presents interactive Plotly charts and a downloadable HTML/PDF report via a FastAPI backend
and Streamlit frontend.

## Rules for agents

- Python 3.11+, type hints, docstrings on public functions, ruff-formatted.
- Never apply scaling/encoding/imputation/SMOTE outside a sklearn/imblearn Pipeline; fit only on training folds.
- All randomness goes through a single seed from config (default 42).
- All models must be compared on identical CV folds.
- Never crash the whole benchmark when one model fails or times out; record status (`ok` | `failed` | `timeout` | `skipped`) and continue.
- Config lives in `configs/*.yaml`, never hardcoded.
- Tests live in `tests/` mirroring `src/acbs`.
- Use logging, not print (CLI tables via rich are allowed).
- Do not add dependencies without adding them to `pyproject.toml`.
- Business logic must not live in API or UI files.

## Commands

```text
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest
ruff check src tests
acbs version
acbs run --dataset breast_cancer --tuning none
uvicorn acbs.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
.\scripts\dev.ps1 test
```

## Definition of done

- Tests covering the change pass.
- No data leakage (transformers fit on train folds only).
- Failed models are recorded, not raised to the UI.
- Config/seed/library versions stored with the experiment.
