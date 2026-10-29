# Auto Classifier Benchmarking System (`acbs v3`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A reproducible, fair benchmarking and recommendation system that compares **8 AutoML frameworks** (AutoGluon, FLAML, GAMA, H2O, LightAutoML, MLJAR, NaiveAutoML, TPOT) and **8 classic ML classifiers** on tabular datasets under unified protocols, answering:

> *"Is AutoML actually worth the compute cost on your dataset, or is plain Random Forest good enough?"*

---

## 🌟 Key Features

1. **Two-Tier Fair Protocol**:
   - **Tier 1 (Classic ML)**: 8 classic models (Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, Gradient Boosting, Naive Bayes, KNN) + `DummyClassifier` baseline evaluated inside strict, leak-free scikit-learn pipelines.
   - **Tier 2 (AutoML Contenders)**: 8 state-of-the-art AutoML frameworks evaluated on raw, unmodified tabular features.
2. **Upload-Only Dataset Workflow**: no built-in datasets — you upload your own CSV/Parquet in the UI, and it is stored under `data/raw/` for the benchmark.
3. **Subprocess Isolation**:
   - Each AutoML framework executes inside its own dedicated virtualenv (`.envs/<framework>/`), preventing PyTorch, NumPy, or scikit-learn version conflicts. A failing or hanging framework is recorded as `failed`/`timeout` and never aborts the run.
4. **Central Scoring Guarantee**:
   - Central scoring by `acbs` from raw prediction probabilities. Eliminates framework self-reporting bias.
5. **AutoML Value Analysis & Recommendation**:
   - Recommends the optimal model based on multi-criteria TOPSIS weighting (Accuracy, Macro-F1, Precision, Recall, ROC-AUC, Train Time, Predict Time, Interpretability, Budget Adherence).
   - Computes empirical $\Delta \text{F1}$ vs. best classic baseline, cost multiplier, and verdict: *"AutoML worth it"*, *"Marginal"*, or *"Not worth it"*.
6. **Plain-HTML Web Frontend** (served by the FastAPI backend):
   - Nine pages: Dashboard, New Benchmark, Run Monitor, Run Detail, Results, History, Datasets, Frameworks, Methodology.
   - Interactive Plotly charts, live reranking without retraining, and HTML/PDF report export.
7. **Deployable Model Export**:
   - One-click `.zip` bundle export containing refit scikit-learn pipelines with `model.joblib` or native AutoML load snippets.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Sathish-hub-61/AutoML-Benchmark-System-ACBS.git
cd AutoML-Benchmark-System-ACBS

# Create virtual environment and install core package
python -m venv .venv
.venv\Scripts\activate   # On Windows (source .venv/bin/activate on Linux/Mac)
pip install -e ".[dev]"
```

### 2. Setup Isolated AutoML Environments
```bash
python scripts/setup_envs.py
python scripts/smoke_test_envs.py   # optional: verify all 8 frameworks import and fit
```

### 3. Launch the System
```bash
uvicorn acbs.api.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser — the backend serves both the API and the web UI. Upload a dataset on the **Datasets** page, then start a benchmark from **New Benchmark**.

An optional Streamlit dashboard is also available (`streamlit run app/streamlit_app.py`, port 8501) against the same API.

---

## 📊 CLI Usage

Run benchmarks directly from your terminal:

```bash
# Classic ML benchmark on a local dataset file
acbs run --dataset data/raw/my_dataset.csv --target label --tuning none

# Two-tier benchmark with FLAML, MLJAR, and AutoGluon at 60s budget
acbs run --dataset data/raw/my_dataset.csv --target label --automl flaml,mljar,autogluon --budget 60

# Batch experiments across datasets
python scripts/run_automl_experiments.py --budget 300 --outer-folds 1
```

---

## 🐳 Docker Deployment

Run the multi-service architecture (PostgreSQL 16 + FastAPI + Streamlit):

```bash
docker-compose up --build
```

> Note: the isolated framework environments are Windows virtualenvs; inside a Linux container the AutoML tier reports frameworks as not installed. See [docs/limitations.md](docs/limitations.md) for platform boundaries.

---

## 📚 Documentation

- [Architecture & Two-Tier Workflow](docs/architecture.md)
- [Benchmark Protocol](docs/protocol.md)
- [Design Decisions](docs/decisions.md)
- [Framework Compatibility Spike](docs/framework_compat.md)
- [Limitations & Platform Boundaries](docs/limitations.md)
- [Reproducibility Guide](docs/reproducibility.md)
- [UI Verification Checklist](docs/ui_checklist.md)

---

## 🧪 Testing

```bash
pytest
ruff check src tests app scripts
```
