# acbs v3: AutoML-Only Benchmark with an AMLB-Style Protocol and an AI Advisor Agent

Plan and IDE-agent prompts, written for your current repo state (Antigravity update of 25 Oct 2026: 28 tests passing, 8 classic models, Optuna, ranking/TOPSIS, FastAPI, a Streamlit prototype, SQLite). The single source of truth for decisions is `ACBS_v3_Master_Design.md`; if this plan and that file disagree, that file wins.

---

## 1. The idea, in one paragraph

Version 2 benchmarked 8 hand-picked scikit-learn models next to the AutoML frameworks. That classic tier is "basic ML" and is now **removed**. Version 3 benchmarks **the 9 AutoML frameworks only**, under a protocol modelled on AMLB (the OpenML AutoML Benchmark): each framework gets raw data, the same folds, seed, cores, memory and metric; scoring is central; failures are results; inference speed is measured; and scores are scaled against a constant-predictor floor. On top sits your original novelty, a **recommendation layer** that ranks frameworks for the user's priorities, and a new **Benchmark Advisor Agent** (a tool-calling LLM) that plans, explains and follows up.

> **acbs v3 benchmarks 9 AutoML frameworks on a user's dataset under an AMLB-style protocol, recommends the best framework for the user's priorities, and lets an AI advisor plan the benchmark and explain the result, with every number grounded in the stored results.**

What is kept from v1: SQLAlchemy storage, FastAPI, min-max and TOPSIS ranking, Pareto and sensitivity analysis, template explanations (the agent's fallback), HTML/PDF reporting, Typer CLI, test structure. What is retired: `models/`, `tuning/`, `preprocessing/` pipelines, imbalance handling (SMOTE), the Streamlit app, and tests that only covered them.

```
                      same folds, seed, cores, memory, metric, one framework at a time
 Dataset ─► Profiler ─► Split (outer stratified K folds)
                          │
                          ▼
   AutoML runner (venv or docker, hard timeout) ─► predictions + cost + inference speed
                          │
                          ▼
   Central scoring ─► scaled vs constant-predictor floor ─► ranks, statistics ─► TOPSIS recommendation
                          │                                                        │
                          ▼                                                        ▼
        Results / reports / AMLB-format CSV                          Advisor Agent (plan, explain)
```

---

## 2. The 9 frameworks (from your notes) and what I verified

Your notes list 10 names because Auto-sklearn and Auto-sklearn 2 are braced together as one framework (that is how AMLB counts 9). You also noted AutoXGBoost and Auto-WEKA as deprecated or not updated. Keep them out and document why. AMLB's own analysis excluded NaiveAutoML (it was evaluated but left out), so here it is labelled **experimental**.

| # | Framework | Search strategy (your notes) | Package | Main budget knob | Variants (fixed "modes") | What I verified |
|---|---|---|---|---|---|---|
| 1 | AutoGluon | Stacked ensembles of preset pipelines | `autogluon.tabular` | `fit(time_limit=...)` | `best_quality`, `high_quality`, `medium_quality` | Release 1.6 supports Python 3.10 to 3.13 |
| 2 | Auto-sklearn | Bayesian opt. of scikit pipelines | `auto-sklearn` | `time_left_for_this_task` | default | **Linux only**, needs C++ compiler. Last PyPI release I found is 0.15.0, so it may not install on Python 3.12 |
| 3 | Auto-sklearn 2 | Bayesian opt. of iterative algorithms | same package: `autosklearn.experimental.askl2.AutoSklearn2Classifier` | same | default | **Do not** `pip install auto-sklearn2`: that PyPI name is a different, third-party simplified re-implementation, not the real thing |
| 4 | FLAML | Cost-frugal optimisation | `flaml` | `fit(time_budget=...)` | default | Simple sklearn-style API |
| 5 | GAMA | Evolutionary opt. of sklearn pipelines | `gama` | `max_total_time` | default | Tabular classification/regression |
| 6 | H2O AutoML | Random search + ensembling | `h2o` | `max_runtime_secs` | default | **Needs Java** (docs list Java 8 to 17) |
| 7 | LightAutoML | Bayesian opt. of linear models and GBMs | `lightautoml` | `timeout` | default | Older reports of Python-version limits; **must be confirmed on your Python** |
| 8 | MLJAR | Custom data-science pipeline | `mljar-supervised` | `total_time_limit` | `Perform`, `Compete` | Mode presets documented |
| 9 | NaiveAutoML (experimental) | Custom data-science pipeline | `naiveautoml` | timeout options (**names changed between versions**) | default | Stops by itself when it thinks nothing better exists, so it may use far less than the budget |
| 10 | TPOT | Evolutionary opt. of sklearn pipelines | `tpot` | `max_time_mins` | default | **API differs between 0.x and newer releases; pin a version and test** |

Following AMLB, frameworks run with **default settings**; only the mode (variant), runtime, cores, memory and target metric are controlled. There are no free hyperparameters in the registry.

Exact Python support for Auto-sklearn, LightAutoML, GAMA, NaiveAutoML and TPOT is not confirmed, so Step B is a real compatibility test rather than an assumption.

---

## 3. Honest risks (plan for these now)

1. **Dependency conflicts.** These frameworks pin different versions of scikit-learn, numpy, pandas and torch. **Solution: one isolated environment per framework, called through a subprocess** (venv now, Docker later). AMLB works the same way.
2. **Auto-sklearn is Linux-only.** On Windows use WSL2 or Docker. If it will not work, record it as "unsupported on this platform", which is a legitimate finding.
3. **Frameworks are time-budget driven.** Training time is a *setting*, not only a *measurement*. Compare *quality at a given budget* and report `budget_used_ratio`.
4. **Frameworks overrun budgets and crash.** Enforce a hard kill timeout (budget + grace) and record `failed` / `timeout`. AMLB shows failures are not random, so never drop them: impute with the constant-predictor score in aggregate and always show the failure count.
5. **Runs are slow.** 9 frameworks x 3 folds x 15 min is about 7 hours per dataset. The UI and agent must show a cost estimate before anything starts.
6. **Different native metrics.** Pass the target metric to the framework (AUC binary, log loss multiclass by default; accuracy, balanced accuracy, macro-F1 selectable). Scoring is still central, from predictions.
7. **Class-order bugs.** Each framework returns probability columns in its own order. A conformance test (Step C4) must catch this.
8. **Meta-learning leakage and hardware gap.** Auto-sklearn 2's meta-models may be optimistic on public datasets; AMLB used 8 vCPU / 32 GB machines and 1 to 4 hour budgets over 104 tasks. Present results as "on this hardware, these datasets, these budgets".
9. **Agent risks.** An LLM can invent numbers or follow instructions hidden in column names. Mitigation in Step H: approval gate, grounding check, schema-validated tools, untrusted-data handling.

Gaps from your notes that this plan turns into features:
- *"Datasets don't include real-time flaws such as missing values"* -> include datasets with missing values and categorical columns, and feed AutoML **raw** data.
- *"Runtime budget is not specified"* -> budget is an explicit, user-visible parameter with a cost estimate and a budget ladder.
- *"New benchmarks designed for AutoML are required"* -> OpenML task/suite intake plus a built-in suite and your own protocol.

---

## 4. Requirements before you start

**Hardware.** AMLB used an AWS m5.2xlarge (8 vCPU, 32 GB). You do not need that, but plan for **8 cores and 16 GB RAM as a practical minimum** (H2O and AutoGluon are memory-hungry) and 30 to 40 GB free disk (AutoGluon pulls in PyTorch).

**Software**
- `uv` (fast per-framework virtual environments) or conda
- Java JDK 11 or 17 (for H2O)
- Linux, WSL2 Ubuntu, or Docker (for Auto-sklearn); a C++ compiler and SWIG on that Linux side
- Python versions available side by side (uv can install 3.9 to 3.12 on demand)
- Docker Desktop (recommended for the final reproducible setup and for enforced cores/memory limits)
- An LLM API key (Claude, OpenAI or Gemini) **or** a local model through Ollama, for the Advisor Agent; the system must still work without it
- An OpenML account is not required to read public tasks; the `openml` Python package is enough

**Decisions to freeze in `docs/protocol.md`:**

| Item | Value |
|---|---|
| Split | Outer stratified K folds, K = 3 by default (1 to 10), seed 42, identical for every framework; AutoML sees only the training part |
| Time budgets | Smoke 60 s, Quick 5 min, Standard 15 min, Thorough 30 min per framework per fold (rankings are consistent across budgets and 30 min tracks 60 min closely, per the AMLB shorter-budget study) |
| Grace | `max(25% of budget, 60 s)`, not told to the framework, for loading, prediction and cleanup |
| Hardware per framework | Same cores (default 4) and memory limit for all; frameworks run **one at a time**; limits enforced by the Docker backend and reported as "passed through" for venv |
| Seed | 42 everywhere |
| Input to AutoML | **Raw** data (missing values, categoricals untouched), no SMOTE, no acbs preprocessing |
| Target metric | AUC (binary) or log loss (multiclass) by default; user may choose accuracy, balanced accuracy, macro-F1. The same metric is given to the framework and used for scoring |
| Scoring | Computed centrally from predictions (never trust framework-reported scores) |
| Floor | Constant predictor (class prior). Scaled score = (score - floor) / (best - floor). Optional random-forest reference toggle, off by default |
| Failures | Statuses `ok`, `failed`, `timeout`, `skipped`, `unavailable`, `unsupported`; failed runs are imputed with the floor in aggregation |
| Inference speed | Rows per second for one in-memory row and for 10,000 rows, median of 10 repeats, measured inside the framework environment |
| Statistics | 3+ datasets: Friedman + Nemenyi (critical-difference diagram). One dataset: fold-wise ranks and Wilcoxon signed-rank. Bradley-Terry trees are a stretch goal |
| Configuration | Vendor defaults and fixed variants only |

---

## 5. What changes in your measurements

| Criterion | How it is measured in v3 |
|---|---|
| Primary metric | Computed by acbs from the framework's predictions on each outer test fold; mean and standard deviation across folds |
| Other metrics | Accuracy, balanced accuracy, macro-F1, AUC, log loss where applicable |
| Scaled score | Relative to the constant-predictor floor (0) and the best framework (1) |
| Fold stability | Standard deviation across folds (a ranking criterion) |
| Training time | Wall-clock fit time **and** `budget_used_ratio` (used / allowed) |
| Inference speed | Rows per second, single-row and 10,000-row (replaces the old latency per 1,000 rows) |
| Memory | Peak RSS of the framework process (psutil) |
| Reliability | Failure and timeout count per framework |
| Interpretability | Not a ranking criterion by default (all frameworks are ensembles or searched pipelines); the framework's own model summary is shown on the Run Detail page |

Headline questions the recommender can answer: **"Which framework fits my priorities?"**, **"How much does the score improve with more budget?"** (budget ladder) and **"How far above the trivial floor is each framework?"**. The old "is AutoML worth it versus classic models" analysis is gone with the classic tier.

---

## 6. Step-by-step plan with IDE-agent prompts

Paste each prompt into Antigravity (or any IDE agent) one at a time. After each one: run `pytest`, `ruff check`, commit.
Start every session with: *"Read AGENTS.md and ACBS_v3_Master_Design.md, and the modules named in the prompt, before changing anything."*

### Step A: Retire the classic tier and housekeeping (Week 1, 1 to 2 days)

**Prompt A1**
```
Read AGENTS.md. One commit per item:
1. Run pytest and ruff. If green, commit existing uncommitted changes. Then create the git
   tag v1-classic so the classic version can always be recovered.
2. Retire the classic tier. Remove the 8 classic models and registry (models/), the tuning
   code (Optuna / RandomizedSearchCV), the acbs preprocessing pipelines, imbalance handling
   (SMOTE, imbalanced-learn) and the Streamlit app (app/) ONLY after the new frontend reaches
   parity (leave app/ for now and mark it deprecated in its README). Remove their tests,
   dependencies in pyproject.toml, and config entries. Keep: storage, API skeleton, ranking
   (min-max, TOPSIS), Pareto, sensitivity, template explanations, reporting, CLI skeleton,
   metrics computation. Keep the metrics module; make _safe_roc_auc_scorer tested for a
   degenerate fold (a class missing in y_true) for binary and multiclass.
3. Replace datetime.datetime.utcnow() with datetime.datetime.now(datetime.UTC) via a shared
   helper utils/time.py.
4. Generate the initial Alembic migration for the schema that remains and run migrations on
   startup instead of Base.metadata.create_all.
5. Add a ConstantPredictor floor in src/acbs/scoring/floor.py (class prior for
   classification). It is a scoring anchor, not a contender, and never appears in the
   framework picker.
Do not add features. Show me the final pytest and ruff output and list everything removed.
```
**Done when:** tests are green, the `v1-classic` tag exists, no classic-model code or dependency remains (except the floor), and `alembic upgrade head` works on a fresh database.

### Step B: Compatibility spike (Week 1, 2 to 3 days). Do this BEFORE writing adapters

**Prompt B1**
```
Goal: find out which of the 9 AutoML frameworks can actually run on this machine, without
touching the main environment.

Create configs/frameworks.yaml with one entry per adapter id: autogluon, autosklearn,
autosklearn2, flaml, gama, h2o, lightautoml, mljar, naiveautoml, tpot. For each: pip
packages (pinned to a version you choose and justify), preferred python_version, system
requirements (e.g. Java for h2o; Linux + C++ compiler + swig for autosklearn), the fixed
variants (modes) allowed, whether it is experimental, and a smoke-test import path. NOTE:
autosklearn2 uses the SAME pip package as autosklearn
(autosklearn.experimental.askl2.AutoSklearn2Classifier). Do NOT install the unrelated PyPI
package named "auto-sklearn2".

Create scripts/setup_envs.py (uses uv) that builds one virtualenv per entry under .envs/<id>/
(gitignored), installing only what that framework needs plus numpy, pandas, pyarrow. Never
install frameworks into the main app environment.

Create scripts/smoke_test_envs.py that, inside each env, fits the framework for ~30-60
seconds on two synthetic datasets: (a) binary, 500 rows, numeric only; (b) 3-class, 800 rows
with missing values and two categorical columns. For each framework record: installed?,
version, fit ok?, predict_proba shape and class order, wall time, and the exact error text.
Write results/compat.json and a readable docs/framework_compat.md table.

Rules: do not spend more than ~30 minutes debugging any single framework; if it cannot be
installed, record the exact error, mark it unsupported on this platform and say whether
WSL2/Docker is the likely fix. Report which frameworks work.
```
**Done when:** `docs/framework_compat.md` exists. Frameworks that pass become "supported". The rest go in a "known limitations" table (this is a result, not a failure).

### Step C: AutoML adapter layer (Weeks 1 to 3)

**Prompt C1: contract, worker and isolated runner**
```
Read docs/framework_compat.md, src/acbs/scoring/ and the metrics module first.

Design and implement the AutoML adapter layer, keeping frameworks isolated from the main app.

1. src/acbs/automl/contracts.py (pydantic):
   AutoMLTask: dataset_id, train_path, test_path (parquet), target, task_type, metric,
     time_budget_s, n_jobs, memory_limit_gb, seed, variant (the fixed mode), input_mode
     ("raw"), fold_index.
   AutoMLOutcome: status ("ok"|"failed"|"timeout"|"skipped"|"unavailable"|"unsupported"),
     reason, framework, variant, framework_version, fit_time_s, budget_s, budget_used_ratio,
     infer_rows_per_s_single, infer_rows_per_s_batch (10,000 rows), n_models_evaluated,
     model_summary (str), leaderboard (list of dicts, optional), peak_mem_mb, classes
     (ordered list), predictions_path (parquet with y_pred and one proba column per class,
     in `classes` order), log_tail.
2. A tiny separate package worker/acbs_worker/ (depends only on numpy, pandas, pyarrow)
   with a CLI: python -m acbs_worker --framework <id> --task task.json --out out_dir.
   It loads train/test, calls adapters/<id>.py (class Adapter with fit(train_df, target,
   task), predict_proba(df), describe()), measures fit time with perf_counter, runs the
   inference benchmark (one in-memory row and 10,000 rows, 10 repeats, median, after one
   warm-up), writes predictions parquet + outcome.json. Any exception -> status "failed"
   with error text.
3. src/acbs/automl/runner.py: run_framework(framework_id, task) -> AutoMLOutcome. Backends
   selectable in config: "venv" (spawn .envs/<id>/bin/python, Scripts\python.exe on Windows)
   now, "docker" in Step J. Enforce a hard timeout = budget + grace (grace = max(25% of
   budget, 60 s), configurable, NOT told to the framework), kill the whole process group on
   timeout, capture the last 200 lines of stdout/stderr, track peak RSS with psutil, and set
   OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS to n_jobs. If the env does not exist ->
   status "unavailable" with the setup command to run.
4. src/acbs/scoring/evaluate.py: given an outcome and y_test, compute ALL metrics centrally
   from the predictions (never the framework's own score), verifying that proba columns
   align with `classes`.
5. A FakeAdapter used only in tests: modes ok / sleep-beyond-budget / crash / bad-class-order.
Tests: ok path, timeout is killed and reported, crash is isolated (main process survives),
missing env gives "unavailable", class-order mismatch is detected. No real framework needed.
```
**Done when:** the fake-adapter tests pass and a timeout really kills the child process.

**Prompt C2: adapters, group 1 (easiest: pure Python, sklearn-like)**
```
Implement worker/acbs_worker/adapters/ for flaml, mljar, gama and tpot. Read each
framework's current docs for exact argument names in the version pinned in
configs/frameworks.yaml before writing code (APIs change between versions; do not guess).
Mapping to honour for every adapter: time budget (seconds), n_jobs, seed, the target metric
from the task (map acbs metric names to the framework's names; if a metric is unsupported
return the closest and record which in the outcome reason), the fixed variant, raw input
with missing values and categoricals handled the way the framework expects (document any
conversion you must do, e.g. cast object columns to category). Do not expose any other
hyperparameters: defaults only.
- flaml: AutoML().fit(task="classification", time_budget, metric, seed, n_jobs); describe()
  returns best_estimator and best_config; leaderboard from search history if available.
- mljar: AutoML(mode=variant ("Perform" or "Compete"), total_time_limit, eval_metric,
  random_state, n_jobs, results_path in a temp dir); leaderboard from its results.
- gama: GamaClassifier(max_total_time, scoring, n_jobs, random_state, output_directory);
  describe() returns the final pipeline string.
- tpot: use the pinned version's API; max time in minutes, n_jobs, random_state; describe()
  returns the fitted pipeline.
For each adapter fill: n_models_evaluated, model_summary, framework_version.
Add an integration test per adapter marked @pytest.mark.automl (skipped if the env is
unavailable) with a 30-second budget on the small synthetic binary dataset.
```

**Prompt C3: adapters, group 2 (heavier or special runtime)**
```
Implement adapters for autogluon, h2o, lightautoml and naiveautoml. Read each framework's
docs for the pinned version first.
- autogluon: TabularPredictor(label, eval_metric, problem_type).fit(train_data, time_limit,
  presets=variant, num_cpus). Variants: best_quality, high_quality, medium_quality. Put its
  model path in a temp dir. leaderboard = predictor.leaderboard.
- h2o: h2o.init(nthreads=n_jobs, max_mem_size from memory_limit_gb); H2OAutoML(
  max_runtime_secs, seed, sort_metric, balance_classes=False). Convert the pandas frame to
  H2OFrame with the target as factor. Always shut the H2O JVM down in a finally block (a
  leaked JVM will pollute the next framework's timing and memory). Check Java is present and
  return status "unavailable" with a clear message if not.
- lightautoml: TabularAutoML(task=Task("binary" or "multiclass"), timeout, cpu_limit,
  general_params / reader_params as needed); handle the class-probability layout carefully
  (binary returns one column).
- naiveautoml (experimental): this framework has no implicit time limit and may stop early.
  Pass the version-appropriate overall timeout argument and candidate timeout (argument
  names changed between versions; read the pinned version's README), and record the real
  time used in budget_used_ratio. Do not pad the run to fill the budget. Mark outcomes
  "experimental" in the stored metadata.
Same conventions as group 1. Add @pytest.mark.automl integration tests.
```

**Prompt C4: adapters, group 3 (Linux-only) and conformance suite**
```
Implement adapters for autosklearn and autosklearn2 (same pip package; autosklearn2 uses
autosklearn.experimental.askl2.AutoSklearn2Classifier). Constraints: Linux only; pass
time_left_for_this_task = budget, per_run_time_limit = a sensible fraction (budget/10, min
10 s), n_jobs, seed, memory_limit from the task. Auto-sklearn spawns worker processes via
dask: make sure the runner kills the whole process group. If the platform is not Linux,
return status "unsupported" with the message "run under WSL2/Docker".
Then write tests/conformance/test_adapters.py that runs EVERY available adapter (skip the
unavailable ones) with a 30 s budget on three datasets: binary numeric, 3-class with
missing values + categoricals, and a 5%-minority imbalanced binary set. It must assert:
(1) outcome.status == "ok"; (2) fit_time <= budget + grace; (3) probabilities sum to ~1 per
row and column order matches `classes`; (4) acbs-computed accuracy matches a manual
sklearn computation on the saved predictions; (5) the inference benchmark fields are
populated and positive; (6) running twice with the same seed gives the same predictions OR
the adapter declares itself non-deterministic in a flag that gets stored. Output a
conformance report to docs/adapter_conformance.md.
```
**Done when:** the conformance report shows every installed framework passing, and failures are listed with reasons.

### Step D: Benchmark engine with the AMLB-style protocol (Weeks 2 to 3)

**Prompt D1**
```
Read ACBS_v3_Master_Design.md Section 3, then src/acbs/automl/ and src/acbs/scoring/.

Build src/acbs/benchmark/engine.py for AutoML-only experiments.
BenchmarkConfig: dataset_id, target, metric (default: auc for binary, logloss for
multiclass; also accuracy, balanced_accuracy, f1_macro), frameworks (list of
framework_id + variant), folds (1 to 10, default 3), budget_s, grace_s, n_jobs, memory_gb,
seed (42), use_rf_reference (default false), early_stopping_variant (default false),
budget_ladder (optional list of budgets).
Rules:
- Create outer stratified folds once per experiment (same seed, identical for every
  framework) and write train/test parquet once per fold. AutoML gets RAW features.
- Run every framework x fold sequentially, never in parallel. A failed/timeout/unavailable
  run never stops the experiment.
- Score centrally. Compute the constant-predictor floor on every fold.
- Aggregation: mean and standard deviation across folds; scaled score = (score - floor) /
  (best - floor) per dataset; failed runs are imputed with the floor in aggregation and the
  failure count is stored and always reported.
- Store per run: framework, variant, framework_version, fold, status, reason, fit_time_s,
  budget_used_ratio, peak_mem_mb, inference rows/s (single and batch), n_models_evaluated,
  model_summary, hardware (cpu count, RAM, OS, python version, versions).
- Budget ladder: if configured, repeat selected frameworks at each budget on fold 0 and
  store score versus budget.
- progress_callback reports framework, fold and phase.
- estimate_cost(config, n_datasets=1) returns wall-clock seconds =
  frameworks x folds x (budget + grace) (+ ladder runs); the CLI/UI/agent must show it first
  and block runs above a configurable limit unless confirmed.
CLI: acbs run --dataset adult --target income --frameworks flaml,mljar --budget 60 --folds 3.
Tests with the FakeAdapter: identical test rows across frameworks, a crashed framework yields
"failed" and is imputed with the floor while all other rows are ok, scaled scores are
correct on a crafted table, estimate_cost arithmetic is correct.
```

### Step E: Statistics and recommendation (Week 4)

**Prompt E1**
```
Read src/acbs/ranking/ and the explain module.

1. Criteria for ranking frameworks: primary metric (mean), fold stability (std, lower is
   better), fit time (log-scaled), inference rows/s (log-scaled, higher is better), peak
   memory, failure rate. Keep min-max and TOPSIS. Presets: balanced, accuracy_first,
   fast_inference, low_resource, reliable. /rerank re-weights without retraining.
2. src/acbs/stats/: average ranks with failure imputation; Friedman test + Nemenyi
   post-hoc and the data for a critical-difference diagram when 3 or more datasets exist;
   with one dataset use fold-wise ranks and a Wilcoxon signed-rank test; a tie statement
   when the top two frameworks are statistically indistinguishable (bootstrap confidence
   intervals on predictions if only one fold). Bradley-Terry trees are a stretch goal: leave
   a clearly marked extension point.
3. Pareto front of score vs training time and of score vs inference speed. Return a
   "family" field per framework (stacking ensemble, Bayesian, evolutionary, cost-frugal,
   custom pipeline) for chart markers.
4. Sensitivity analysis: rank shifts when weights change.
5. Template explanations (the agent's fallback) using configs/frameworks_info.yaml with each
   framework's strategy in one line (AutoGluon = stacked ensembles of preset pipelines,
   Auto-sklearn = Bayesian optimisation of sklearn pipelines, Auto-sklearn 2 = Bayesian
   optimisation of iterative algorithms, FLAML = cost-frugal optimisation, GAMA =
   evolutionary optimisation of sklearn pipelines, H2O = random search + ensembling,
   LightAutoML = Bayesian optimisation of linear models and GBMs, MLJAR and NaiveAutoML =
   custom data-science pipelines, TPOT = evolutionary optimisation). Include caveats:
   budget used, failures, non-determinism, slow inference, experimental flag, Auto-sklearn 2
   meta-learning caveat.
6. Budget ladder analysis: score versus budget per framework, flagging diminishing returns.
Tests: crafted tables for ranks, ties, imputation, presets that change the winner (speed
weights pick a faster framework), and a failed framework ranking below the floor-imputed
ones.
```

### Step F: Storage, API and dataset intake (Week 4)

**Prompt F1**
```
Read src/acbs/storage/ and src/acbs/api/.
1. New Alembic migration: tables datasets, experiments (config JSON, metric, folds, budget,
   seed, hardware_json), runs (experiment, framework, variant, fold, status, reason,
   fit_seconds, budget_used_s, peak_mem_mb, infer_rows_per_s_single,
   infer_rows_per_s_batch, model_summary, leaderboard_json, log_tail, artifacts_path),
   predictions (run, row index, y_true, y_pred, proba), scores (run, metric, value),
   framework_versions. Agent tables come in Step H.
2. Move all API routes under /api. Serve the static web/ directory from FastAPI at the same
   origin. Report every route moved.
3. Endpoints: GET /api/frameworks (id, name, strategy, variants, installed?, version,
   os_supported, experimental, reason if unavailable); POST /api/experiments;
   GET /api/experiments/{id}/status (per framework x fold state and elapsed);
   POST /api/estimate; POST /api/experiments/{id}/rerank (no retraining);
   GET /api/experiments/{id}/figures/{name} returning Plotly figure JSON built from
   charts.py; DELETE/cancel that kills the running framework subprocess;
   GET /api/experiments/{id}/export?format=amlb_csv|pdf|predictions.
4. AMLB-format export: results.csv with columns id, task, fold, framework, constraint,
   result, metric, duration, seed (+ our extra columns after them); metrics that are
   lower-is-better are negated and prefixed neg_ as AMLB does.
5. Dataset intake: upload CSV/Parquet; OpenML task or suite ID through the openml package;
   a small built-in suite. Validate and profile every dataset; reject unsafe or unusable
   data with plain-language reasons; never execute anything from a dataset.
6. Jobs: experiments can run for hours. Run them in a separate process with persisted
   progress so an API restart marks the experiment "interrupted" instead of losing it.
Update API tests (use the FakeAdapter via a config flag so tests do not need real frameworks).
```

### Step G: HTML/CSS/JS frontend with Plotly charts (Week 5 to 6)

This also finishes your "Interactive Charts Missing" gap from Antigravity.

**Prompt G1**
```
Read app/streamlit_app.py (reference only), ACBS_Project_Pages_Blueprint, and
src/acbs/reporting/charts.py. Build a plain HTML, CSS and JavaScript frontend (ES modules,
no framework, no build step) under web/: one HTML file per page (index, datasets,
new-benchmark, run-monitor, results, model-detail, history, frameworks, methodology),
web/css/ (tokens.css, base.css, components.css), web/js/api.js (the only module that calls
fetch), web/js/shell.js, web/js/charts.js, web/js/components/ (experiment-form,
framework-picker, chart-frame, metrics-table, recommendation-card) and web/js/pages/.
Keep the UI free of ML logic. Render Plotly figure JSON from the backend with a vendored
Plotly.js. Use textContent (never innerHTML with untrusted text), a strict
Content-Security-Policy with no inline scripts, polling with AbortController that pauses
when the tab is hidden, and Playwright for one end-to-end test. Remove the Streamlit app
only after the new pages reach parity.

New Benchmark page: dataset profile card; target; metric picker; framework picker listing
each AutoML framework with variant selector, installed/not-installed/unsupported badge and
reason, its one-line strategy, an "experimental" tag for NaiveAutoML and a checkbox; folds
(1 to 10); budget presets (Smoke 1 / Quick 5 / Standard 15 / Thorough 30 min, or custom);
cores and memory; optional toggles (budget ladder, early-stopping variant,
random-forest reference); and a live cost estimate ("about 6 h 45 min"). Block runs whose
estimate exceeds a configurable limit unless the user confirms. Advisor panel docked on
the right.

Run Monitor: grid of framework x fold cells with status and elapsed-vs-budget bars.

Results page, recommendation card on top (winner, explanation, stability, failure count)
and tabs:
 - Overview: scaled-score bars + Pareto scatter (score vs training time), marker shape by
   framework family
 - Rankings: average ranks; critical-difference diagram with 3+ datasets, Wilcoxon table
   with one dataset
 - Time & cost: training time, budget used, peak memory
 - Inference: rows per second, single-row and 10,000-row (log scale)
 - Diagnostics: confusion matrix, ROC and calibration for the winner and a selectable run
 - Sensitivity: tornado / rank-shift chart under weight changes
 - Budget ladder: score versus budget
 - Failures: status and reason per framework and fold
Weight sliders call /rerank and refresh without retraining. Export buttons: PDF, AMLB-format
results.csv, predictions.
Add a manual UI test checklist in docs/ui_checklist.md.
```

### Step H: Benchmark Advisor Agent (Week 6)

**Prompt H1: agent core**
```
Read ACBS_v3_Master_Design.md Section 5 and src/acbs/services/.

Build src/acbs/agent/ for the Benchmark Advisor Agent.
1. llm.py: an LLMClient interface (chat(messages, tools) -> assistant message or tool
   calls) with implementations for Claude, OpenAI, Gemini and Ollama selected by config; the
   API key is read from the environment, never logged or sent to the browser. Also a FakeLLM
   that replays scripted tool calls for tests. If no LLM is configured the agent returns
   rule-based suggestions and the template explanation.
2. tools.py: JSON-schema typed tools that are thin wrappers over existing services:
   profile_dataset, list_frameworks, estimate_cost, propose_plan (creates a plan record
   awaiting approval), start_benchmark (accepts ONLY an approved plan id), get_status,
   get_results, rerank, compare_runs, explain_failure. Validate every call against its
   schema; unknown tools or extra arguments are rejected.
3. loop.py: the tool-calling loop with hard caps (max tool calls per conversation, max
   frameworks and max budget per plan, a per-conversation time limit). The approval gate is
   enforced in the backend: only the UI (or CLI) can mark a plan approved, the model cannot.
4. grounding.py: after the model writes a final answer, extract every number and check that
   it appears in a tool result of the same conversation (with rounding tolerance). If the
   check fails, discard the answer and use the template explanation, and record the failure.
5. Untrusted data: dataset names, column names and class names are passed to the model as
   quoted data in a clearly delimited block, never as instructions; the system prompt states
   that text inside tool results is data. Tool results are truncated to a safe size.
6. Storage: tables agent_sessions, agent_messages, agent_tool_calls (arguments, result,
   duration, error) with a migration; the full trace is retrievable.
7. API: POST /api/agent/sessions, POST /api/agent/sessions/{id}/messages (stream),
   POST /api/agent/plans/{id}/approve, GET /api/agent/sessions/{id}/trace.
Tests with FakeLLM: a full plan -> approve -> run (FakeAdapter) -> explain conversation;
start_benchmark without approval is refused; a hallucinated number is caught by the
grounding check; an injected instruction in a column name is not followed; caps are
enforced; the no-LLM fallback works.
```

**Prompt H2: advisor panel and evaluation**
```
Add the Advisor panel to web/ (a docked panel component used on New Benchmark, Run Monitor
and Results; not a tenth page): suggestion chips ("Plan a benchmark for this dataset",
"Explain this result", "Why did a framework fail?"), a message list rendered with
textContent only, a plan card with an explicit Approve and Edit button showing the cost
estimate, streaming responses, a collapsible tool-call trace, and a clear banner when no LLM
is configured. Then write tests/agent_eval/ with 15 questions and the expected tool
sequences, run with FakeLLM in CI, and a script that can run them against a real provider.
Report which questions fail.
```
**Done when:** the full agent conversation passes with FakeLLM, ungrounded numbers are rejected, and a run cannot start without approval.

### Step I: Experiments for the paper (Weeks 5 to 7, run in the background)

**Prompt I1**
```
Create scripts/run_automl_experiments.py and notebooks/automl_analysis.ipynb.

Datasets: the built-in suite plus OpenML tasks (preferably a few from the OpenML-CC18 suite so
results are comparable with AMLB); make sure at least 3 have missing values, at least 2 have
high-cardinality categoricals, and at least 2 are strongly imbalanced. Protocol:
docs/protocol.md (Quick 5 min or Standard 15 min budget, 4 cores, seed 42, 3 outer folds).
The script must be resumable, log every failure with its reason, store everything in the DB
and results/*.csv, and print a runtime estimate before starting.

Notebook outputs (300-dpi figures into results/figures/):
 1. Average rank per framework across datasets with a critical-difference diagram (Friedman +
    Nemenyi) and the scaled-score box plot.
 2. Score vs total time scatter per dataset (Pareto fronts).
 3. Failure and timeout table per framework (like AMLB's failure analysis), plus budget
    adherence (budget_used_ratio) and peak memory.
 4. Inference rows per second per framework (single-row and 10,000-row, log scale).
 5. Budget ladder: score versus budget on 3 datasets.
 6. How the recommended framework changes with user priorities (balanced / accuracy_first /
    fast_inference / low_resource / reliable): this supports our novelty claim.
 7. Imbalanced datasets: accuracy vs balanced accuracy / macro-F1 divergence.
 8. Optional: early-stopping variant vs default for frameworks that support it, reported
    separately.
Also write docs/limitations.md: hardware (compare with AMLB's 8 vCPU / 32 GB), versions,
budget-driven rankings, platform limits, Auto-sklearn 2 meta-learning caveat, NaiveAutoML
experimental status, frameworks that could not run and why.
```

### Step J: Docker, tests, security, docs and review material (Week 7)

**Prompt J1**
```
1. Docker: a base image for the app (API serving the static web/ frontend + PostgreSQL via
   docker-compose; no separate UI container) and one Dockerfile per framework environment
   under docker/frameworks/<id>/ (Auto-sklearn on a Linux base, H2O with a JDK installed).
   The runner must support the "docker" execution backend in addition to "venv", selected in
   config; the docker backend enforces cores and memory limits and the report states which
   limits are enforced.
2. Tests: unit tests for scoring, ranking, statistics, runner, agent; integration tests with
   the FakeAdapter and FakeLLM; real-framework tests behind @pytest.mark.automl; a
   Playwright end-to-end journey (upload, configure, approve plan, run with fake adapters,
   monitor, results, export).
3. Security review: file upload validation and size limits, no code execution from datasets,
   XSS (autoescaping in reports, textContent in the frontend, strict CSP), prompt injection
   through column names, secrets only in environment variables, optional API token handling
   in sessionStorage, CORS, rate limiting on the agent endpoint, subprocess argument safety
   in the runner, path traversal in artifact downloads.
4. README.md and docs/architecture.md: architecture diagram (mermaid), the adapter contract,
   the AMLB-style protocol, how to add a new framework (one adapter file + one entry in
   configs/frameworks.yaml), the agent design and guardrails, known limitations.
5. docs/reproducibility.md: seeds, versions, hardware recorded per run, exact commands to
   reproduce each figure, and the honest note that time-bounded search is not bit-for-bit
   repeatable.
```

### Step K: Model export (optional, only if time remains)

**Prompt K1**
```
Add export of the recommended framework's model. Ask the adapter to save the framework's
native artifact (AutoGluon predictor folder, H2O MOJO, MLJAR results folder, pickled
sklearn-style pipeline for FLAML/GAMA/TPOT where supported) into a zip with a README that
states which environment and framework version is needed to load it. Expose
GET /api/experiments/{id}/export?format=model and a Download button. Be explicit in the UI
that artifacts need the same framework version to load.
```

---

## 7. Timeline (7 weeks, adjust to your real deadline)

| Week | Steps | Output |
|---|---|---|
| 0 | A (tag `v1-classic`), B | Compatibility spike for all 9 frameworks on your OS and Python versions; decide Docker or WSL2 |
| 1 | A, C1 | Classic tier retired, contract, registry, runner (venv), FakeAdapter, constant-predictor floor, `/api` move |
| 2 | C2, C3, D1 (start) | Adapters for AutoGluon, FLAML, H2O, MLJAR, LightAutoML (the easy five); central scoring; inference benchmark |
| 3 | C3, C4, D1 | Adapters for Auto-sklearn (+2), GAMA, TPOT, NaiveAutoML; failure handling; Docker backend started |
| 4 | E1, F1 | Aggregation, ranking, statistics, budget ladder; AMLB-format export; OpenML intake |
| 5 | G1 (first 5 pages), start I1 | Frontend foundation and the first 5 pages; experiments begin in the background |
| 6 | G1 (remaining pages and charts), H1, H2 | Remaining pages, charts, Advisor Agent (tools, guardrails, panel) |
| 7 | J1, finish I1, paper | Test, security review, E2E, performance, Docker, documentation, demo, report |

**Safety milestone (end of week 3):** `acbs run --dataset adult --target income --frameworks flaml,mljar,autogluon --budget 60 --folds 3` prints one ranked table of AutoML frameworks with scaled scores and failure counts. If that works, everything after it is polish.

**Cut order if time runs short:** Bradley-Terry trees, early-stopping variant, Singularity/AWS runners, random-forest reference, extra Wilcoxon tables, then the budget ladder. **Never cut:** scoring correctness, failure isolation, the fair-run rules, and the agent's approval gate.

**Start the experiments (I1) as early as week 4 in the background.** They take hours and they are what your paper needs.

---

## 8. Experiment cost estimator

Total time is roughly `datasets x frameworks x folds x (budget + grace)`, because frameworks run one at a time. (Grace is only used when a framework overruns; typical runs are close to the budget.)

| Scenario | Frameworks | Budget | Datasets | Folds | Approx. wall time |
|---|---|---|---|---|---|
| Smoke check | 9 | 1 min | 5 | 1 | ~45 min |
| Quick | 9 | 5 min | 1 | 3 | ~2.3 h |
| Standard | 9 | 15 min | 1 | 3 | ~6.8 h |
| Main experiment | 9 | 5 min | 8 | 3 | ~18 h |
| Budget ladder | 5 | 5/15/30 min | 3 | 1 | ~12.5 h |

(Add about 25% for overruns and failures.) Run them overnight and keep the machine otherwise idle so timings are clean.

---

## 9. What to say in your next review

- **Problem:** choosing among AutoML frameworks is hard; benchmarks like AMLB compare them but do not recommend, and they are not interactive.
- **Contribution:** acbs benchmarks 9 AutoML frameworks under an AMLB-style protocol, **recommends** the best framework for the user's priorities, exports AMLB-compatible results, and adds an AI advisor that plans and explains benchmarks with every number grounded in stored results.
- **Method:** isolated per-framework environments, raw data, central scoring, constant-predictor floor, outer folds, failure imputation, inference speed measurement, statistical tests, budget ladder.
- **Honest limitation:** results are specific to the budget, hardware, and framework versions used, and some frameworks may not run on every platform. AMLB used 8 vCPU / 32 GB cloud machines and 1 to 4 hour budgets over 104 tasks; this project uses far smaller evidence and says so.

---

## 10. Working tips for the IDE agent

1. One prompt = one step. Do not merge steps B and C: the compatibility spike decides what is feasible.
2. Tell the agent to read each framework's documentation for the pinned version before writing its adapter. Several APIs (TPOT, NaiveAutoML, LightAutoML) have changed between releases.
3. When a framework fails to install, ask the agent to record the exact error and move on; documenting a failure is a result.
4. After every step: *"Review what you just wrote for data leakage, unhandled exceptions, hardcoded values, prompt-injection risks and missing tests. List issues, then fix them."*
5. Never let the agent install an AutoML framework into the main app environment.
6. Never let the agent compute or "fix" scores in the LLM layer: the agent only reads numbers from tools.

Sources: [AMLB site](https://openml.github.io/automlbenchmark/), [openml/automlbenchmark](https://github.com/openml/automlbenchmark), [AMLB paper (JMLR)](https://jmlr.org/papers/v25/22-0493.html), [shorter time constraints and early stopping](https://arxiv.org/html/2504.01222v1).
