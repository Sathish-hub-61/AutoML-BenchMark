# Reproducibility Guide

This guide describes how to replicate all empirical results, tables, and figures reported in the AutoML Benchmarking study using `acbs v2`.

---

## 1. System & Hardware Specifications

All benchmark experiments in the reference paper were evaluated under the following baseline environment:

- **CPU**: 8-core x86_64 Processor (4 threads allocated per contender via `n_jobs=4`)
- **Memory**: 16 GB Physical RAM (SWAP enabled)
- **Operating System**: Windows 11 Enterprise / Ubuntu 22.04 LTS (WSL2 & Docker)
- **Python**: Python 3.12 (Base App), Python 3.10 / 3.11 (Framework environments)
- **Global Seed**: `42` (fixed across train/test splits, fold indices, and model initializations)

---

## 2. Setting Up Virtual Environments

Run the automated virtual environment builder to install isolated dependencies for all verified frameworks:

```bash
python scripts/setup_envs.py
```

To run smoke-tests confirming environment readiness:

```bash
python scripts/smoke_test_envs.py
```

---

## 3. Running Benchmark Experiments

### Main Experiment (Standard 300s Budget)
Executes all classic classifiers and AutoML contenders across 13 registry datasets:

```bash
python scripts/run_automl_experiments.py --budget 300 --outer-folds 1 --output-csv results/automl_benchmark_results.csv
```

### Quick Exploration (60s Smoke Check)
```bash
python scripts/run_automl_experiments.py --budget 60 --tier quick
```

### Variance Study (3 Outer Folds)
```bash
python scripts/run_automl_experiments.py --budget 300 --outer-folds 3 --datasets breast_cancer,wine,adult
```

---

## 4. Reproducing Publication Figures

Launch Jupyter Notebook to execute `notebooks/automl_analysis.ipynb`:

```bash
jupyter notebook notebooks/automl_analysis.ipynb
```

Executing all cells generates 300-DPI publication figures in `results/figures/`:

| Figure File | Description | Section |
|---|---|---|
| `01_average_ranks.png` | Average rank across datasets & Critical Difference bars | Section 1 |
| `02_pareto_quality_vs_time.png` | Pareto frontier: Macro-F1 vs. Wall Time ($\log$) | Section 2 |
| `03_reliability_and_memory.png` | Execution status distribution (%) and Peak Memory RSS | Section 3 |
| `04_inference_latency.png` | Prediction latency per 1,000 rows ($\log$) | Section 4 |
| `05_automl_value_verdicts.png` | $\Delta \text{Macro-F1}$ (Best AutoML minus Best Classic) & ROI verdict | Section 5 |

---

## 5. Launching the Interactive Platform

Start the FastAPI backend and Streamlit dashboard locally:

```bash
# Terminal 1: API Backend
uvicorn acbs.api.main:app --reload --port 8000

# Terminal 2: Streamlit Dashboard
streamlit run app/streamlit_app.py
```

Or deploy using Docker Compose:

```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8501`.
