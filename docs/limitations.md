# Limitations & Platform Boundaries

This document details hardware dependencies, platform boundaries, library version pins, and known trade-offs documented during the `acbs` AutoML benchmark implementation.

---

## 1. Operating System & Platform Boundaries

- **Auto-sklearn & Auto-sklearn 2**:
  - Requires Linux, GNU C++ toolchain, and SWIG.
  - On Windows 10/11 and macOS, these frameworks are recorded with status `unsupported` and message `"Linux only (requires C++ compiler and SWIG). Run under WSL2 or Docker."`.
  - For reproduction of Linux-only frameworks, use the Docker container setup (`docker/frameworks/autosklearn/Dockerfile`).

- **H2O AutoML**:
  - Requires a compatible Java Runtime Environment (JDK 8, 11, or 17).
  - If Java is absent on the host PATH, H2O will fail JVM startup and register status `unavailable`.

---

## 2. Library Version Compatibility

- **TPOT**:
  - Newer versions of `setuptools>=70` deprecated distutils internals required by TPOT's code generator. Environments running TPOT pin `setuptools<70`.
- **NaiveAutoML**:
  - Upstream release 0.1.6 contains an internal heuristic calculation with missing-value handling on Windows platforms (`KeyError: nan` in scikit-learn pipeline search). When this occurs, status `failed` is recorded with full stacktrace.
- **LightAutoML & AutoGluon**:
  - AutoGluon requires PyTorch and substantial disk allocation (~2-4 GB per virtual environment).
  - LightAutoML pins specific versions of `poetry-core` and scikit-learn.

---

## 3. Hardware & Concurrency Bounds

- **Memory Ceiling**:
  - Certain frameworks (notably AutoGluon with multi-layer neural network stacking and H2O) require at least 8 GB RAM per run. On machines with $\le 8\text{ GB}$ RAM, lower the concurrent dataset budget or reduce `n_jobs`.
- **Budget-Driven Training Time**:
  - AutoML frameworks are configured by time budgets rather than stopping criteria alone. Thus, training time is a resource *allocation*, and framework performance must be scored by quality achieved within the allowed budget and `budget_used_ratio`.
- **Inference Latency Trade-Off**:
  - Multi-layer stacked ensembles (AutoGluon, H2O, MLJAR Ensemble) yield high predictive metrics but can exhibit $10\times$ to $100\times$ slower inference latency compared to simple Logistic Regression or Random Forest models. User preferences favoring fast inference automatically apply penalty discounts to deep ensembles.
