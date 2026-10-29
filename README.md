# Auto Classifier Benchmarking System (`acbs v2`)

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
2. **Subprocess & Docker Isolation**:
   - Each AutoML framework executes inside its own dedicated virtualenv (`.envs/<framework>/`) or Docker container. Prevents PyTorch, NumPy, or scikit-learn version conflicts.
3. **Central Scoring Guarantee**:
   - Central scoring by `acbs` from raw prediction probabilities. Eliminates framework self-reporting bias.
4. **AutoML Value Analysis & Recommendation**:
   - Recommends the optimal model based on multi-criteria TOPSIS weighting (Accuracy, Macro-F1, Precision, Recall, ROC-AUC, Train Time, Predict Time, Interpretability, Budget Adherence).
   - Computes empirical $\Delta \text{F1}$ vs. best classic baseline, cost multiplier, and verdict: *"AutoML worth it"*, *"Marginal"*, or *"Not worth it"*.
5. **Interactive Dashboard & Plotly Charts**:
   - Streamlit frontend with Grouped Metric Bars, Accuracy vs. Log(Time) Pareto Frontiers, Radar charts, Inference Latency analysis, and Framework Leaderboard inspection.
   - Live reranking without retraining.
6. **Deployable Model Export**:
   - One-click `.zip` bundle export containing refit scikit-learn pipelines with `model.joblib` or native AutoML load snippets.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Sathish-hub-61/AutoML-BenchMark.git
cd "AutoML BenchMark"

# Create virtual environment and install core package
python -m venv .venv
.venv\Scripts\activate   # On Windows (source .venv/bin/activate on Linux/Mac)
pip install -e ".[dev]"
```

### 2. Setup Isolated AutoML Environments
```bash
python scripts/setup_envs.py
```

### 3. Launch Interactive Services
```bash
# Terminal 1: FastAPI Backend
uvicorn acbs.api.main:app --reload --port 8000

# Terminal 2: Streamlit Dashboard
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 CLI Usage

Run benchmarks directly from your terminal:

```bash
# Classic ML benchmark on Wisconsin Breast Cancer dataset
acbs run --dataset breast_cancer --tuning none

# Two-tier benchmark with FLAML, MLJAR, and AutoGluon at 60s budget
acbs run --dataset adult --automl flaml,mljar,autogluon --budget 60

# Batch experiments across all registry datasets
python scripts/run_automl_experiments.py --budget 300 --outer-folds 1
```

---

## 🐳 Docker Deployment

Run the complete multi-service architecture (PostgreSQL 16 + FastAPI + Streamlit):

```bash
docker-compose up --build
```

---

## 📚 Documentation

- [Architecture & Two-Tier Workflow](docs/architecture.md)
- [Benchmark Protocol](docs/protocol.md)
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
