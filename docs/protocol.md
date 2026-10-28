# AutoML Benchmark Evaluation Protocol

This document establishes the experimental protocol for comparing classic machine learning classifiers and automated machine learning (AutoML) frameworks in `acbs v2`.

---

## 1. Experimental Splits & Seed

- **Evaluation Split**: Stratified 80/20 train/test hold-out for interactive benchmarking; 3-fold repeated stratified outer folds for research mode experiments.
- **Random Seed**: Single global seed `42` across all splits, baseline initializations, and framework random states.
- **Fairness Guarantee**: Every candidate (classic classifier or AutoML framework) is evaluated on the exact same test fold indices.

---

## 2. Input Data Protocol

- **Classic Tier**: Feeds through a strict leak-free scikit-learn / imblearn `Pipeline` (imputation, one-hot encoding, standard scaling, and optional SMOTE) fit solely on the training fold.
- **AutoML Tier**: Receives **raw** tabular data (missing values, numericals, and categoricals untouched). No manual SMOTE or feature engineering is applied to AutoML contenders, as native preprocessing is part of what AutoML benchmarks evaluate.

---

## 3. Execution & Isolation

- **Runtime Isolation**: Each AutoML contender executes inside an isolated virtual environment (`.envs/<framework_id>/`) invoked via a subprocess worker (`acbs.automl.workers.<id>_worker`).
- **Sequential Execution**: Frameworks run strictly one-by-one to prevent CPU cache contention, memory competition, or timing distortion.
- **Concurrency & Threads**: All contenders are allocated identical CPU resources (`n_jobs=4` default, setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`).
- **Hard Timeout Enforcement**: Subprocess execution is terminated with process-group kill upon exceeding `budget + grace` (grace period = $\max(0.25 \times \text{budget}, 60\text{s})$).

---

## 4. Budgets & Configurations

| Tier | Budget | Target Usage |
|---|---|---|
| **Quick** | 60 seconds | Smoke testing, fast interactive exploration, responsive debugging |
| **Standard** | 300 seconds | Main benchmark experiments, fair comparison across typical constraints |
| **Extended** | 900 seconds | Variance studies, deep ensemble convergence, research papers |

- **Presets & Hyperparameters**: Vendor recommended defaults and near-intended usage (e.g. `mode="Perform"` for MLJAR, `medium_quality` for AutoGluon).

---

## 5. Central Metric Evaluation

To ensure rigorous and unpolluted metric comparisons:
1. Frameworks only output out-of-sample predictions (`y_pred` and predicted probabilities `y_proba` matching alphabetical class ordering).
2. `acbs.automl.evaluate` computes all reported metrics centrally:
   - **Accuracy**
   - **Balanced Accuracy**
   - **Precision (macro)**
   - **Recall (macro)**
   - **F1 (macro)**
   - **ROC-AUC (OVR, macro)**
   - **Log-Loss**
   - **Wall-clock Training Time (s)**
   - **Inference Latency per 1,000 samples (s)**
   - **Budget Used Ratio** ($\text{budget\_used} / \text{budget\_allowed}$)
   - **Peak Memory RSS (MB)**
3. Baseline majority classifier (`DummyClassifier(strategy="most_frequent")`) is evaluated on every dataset to ground-truth normalisation.

---

## 6. Execution Statuses

- `ok`: Finished within budget and generated valid predictions.
- `failed`: Framework raised an unrecoverable exception or error during execution.
- `timeout`: Framework exceeded budget + grace period and was terminated by runner.
- `unavailable`: Environment not found or prerequisite (e.g., Java for H2O) missing.
- `unsupported`: Framework cannot run on current operating system (e.g. Auto-sklearn on Windows).
- `skipped`: Intentionally omitted by user configuration.
