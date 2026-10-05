# Frozen decisions (week 1)

1. **Frontend:** FastAPI backend + Streamlit MVP.
2. **Database:** SQLite in development; PostgreSQL in Docker; SQLAlchemy 2.0.
3. **Hyperparameter search:** Optuna TPE with a trial budget; RandomizedSearchCV as an alternative.
4. **Evaluation protocol:** Stratified 5-fold CV, fixed seed, identical folds, 80/20 hold-out for final scores.
5. **Multiclass metrics:** Macro precision/recall/F1; ROC-AUC `ovr` + `macro`; also weighted variants.
6. **Timing:** Training time = wall-clock `fit()`; prediction time = median seconds per 1,000 rows after a warm-up; `n_jobs` fixed and recorded.
7. **Ranking:** Weighted sum of min-max scores, TOPSIS, and a Pareto front. Template explanations (no LLM).
8. **Scope:** Tabular binary and multiclass classification only.

Windows notes: `scripts/dev.ps1` is the primary local tooling; process-based timeouts are used because `SIGALRM` is unavailable.
