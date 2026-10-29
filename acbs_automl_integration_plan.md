# acbs v2: Integrating the 9 AutoML Frameworks

Plan and IDE-agent prompts, written for your current repo state (Antigravity update of 25 Oct 2026: 28 tests passing, 8 classic models, Optuna, ranking/TOPSIS, FastAPI, Streamlit, SQLite).

---

## 1. The idea, in one paragraph

Right now acbs benchmarks **8 hand-picked scikit-learn classifiers**. That is "basic ML". Your own research (AMLB, your handwritten notes, your paper's introduction) is about **AutoML frameworks**, and your research-gap slide says existing AutoML benchmarks *compare frameworks but stop short of recommending one*.

So the project becomes:

> **acbs v2 benchmarks AutoML frameworks (plus your 8 classic models as baselines) on a user's dataset under one fair protocol, then recommends the best option for that user's priorities, including "plain Random Forest is good enough, don't use AutoML".**

Nothing you built is wasted. The 8 classic models become the **baseline tier**, the metrics/timing/ranking/TOPSIS/Pareto/report code is reused as is, and the AutoML frameworks are a new **contender tier** that plugs into the same pipeline.

```
                     ┌───────────── same hold-out split, seed, CPU, metrics ─────────────┐
 Dataset ──► Profiler ─┤  TIER 1: Classic baselines (existing 8 models + Dummy)             │
                       │  TIER 2: AutoML contenders (9 frameworks, fixed time budget)       ├─► Central evaluation ─► Ranking/TOPSIS/Pareto ─► Recommendation + "is AutoML worth it?" ─► Reports/UI
                       └────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 9 frameworks (from your notes) and what I verified

Your notes list 10 names because Auto-sklearn and Auto-sklearn 2 are braced together as one framework (that is how AMLB counts 9). You also noted AutoXGBoost and Auto-WEKA as deprecated/not updated. Keep them out and document why.

| # | Framework | Search strategy (your notes) | Package | Main budget knob | What I verified |
|---|---|---|---|---|---|
| 1 | AutoGluon | Stacked ensembles of preset pipelines | `autogluon.tabular` | `fit(time_limit=...)` | Release 1.6 supports Python 3.10 to 3.13 |
| 2 | Auto-sklearn | Bayesian opt. of scikit pipelines | `auto-sklearn` | `time_left_for_this_task` | **Linux only**, needs C++ compiler. Last PyPI release I found is 0.15.0, so it may not install on Python 3.12 |
| 3 | Auto-sklearn 2 | Bayesian opt. of iterative algorithms | same package: `autosklearn.experimental.askl2.AutoSklearn2Classifier` | same | **Do not** `pip install auto-sklearn2`: that PyPI name is a different, third-party simplified re-implementation (no meta-learning), not the real thing |
| 4 | FLAML | Cost-frugal optimisation | `flaml` | `fit(time_budget=...)`, `estimator_list` | Simple sklearn-style API |
| 5 | GAMA | Evolutionary opt. of sklearn pipelines | `gama` | `max_total_time` | Tabular classification/regression; optimises log-loss by default |
| 6 | H2O AutoML | Random search + ensembling | `h2o` | `max_runtime_secs` | **Needs Java** (a JDK/JRE). Docs I found list Java 8 to 17 |
| 7 | LightAutoML | Bayesian opt. of linear models and GBMs | `lightautoml` | `timeout` | Older reports of Python-version limits; **must be confirmed on your Python** |
| 8 | MLJAR | Custom data-science pipeline | `mljar-supervised` | `total_time_limit`, `mode` (Explain / Perform / Compete / Optuna) | Mode presets documented |
| 9 | NaiveAutoML | Custom data-science pipeline | `naiveautoml` | timeout options (**names changed between versions**) | Stops by itself when it thinks nothing better exists, so it may use far less than the budget |
| 10 | TPOT | Evolutionary opt. of sklearn pipelines | `tpot` | `max_time_mins` | **API differs between 0.x and newer releases; pin a version and test** |

I could not confirm exact Python support for Auto-sklearn, LightAutoML, GAMA, NaiveAutoML and TPOT from what I found, so step B below is a real compatibility test rather than an assumption.

---

## 3. Honest risks (plan for these now)

1. **Dependency conflicts.** These frameworks pin different versions of scikit-learn, numpy, pandas, and torch. They will not all install in one environment next to your Python 3.12 app. **Solution: one isolated virtual environment per framework, called through a subprocess.** This is also how AMLB works (each framework is installed and run in isolation).
2. **Auto-sklearn is Linux-only.** On Windows or macOS use WSL2 or Docker for that one framework. If it will not work, record it as "unsupported on this platform", which is a legitimate finding.
3. **Frameworks are time-budget driven.** They run until the budget is spent, so training time is a *setting*, not a *measurement*. Your "training time" criterion must change meaning for AutoML (see Section 5).
4. **Frameworks overrun budgets and crash.** Use a hard kill timeout and record `failed` / `timeout`. AMLB reports framework failures as a result, so you should too.
5. **Runs are slow.** 9 frameworks x 5 min = 45 min per dataset per run. Build a cost estimator into the UI (Section 8).
6. **Different native metrics.** Not every framework can optimise macro-F1. You will score everything centrally from predictions, so the *comparison* is fair even if the *optimisation target* differs (document the target used per framework).
7. **Class-order bugs.** Each framework returns probability columns in its own order. A conformance test (step C4) must catch this.
8. **AutoML ensembles are hard to interpret.** Your interpretability criterion needs a score for them (low, 1 to 2).

Gaps from your notes that this plan turns into features:
- *"Datasets don't include real-time flaws such as missing values"* → include datasets with missing values and categorical columns, and feed AutoML **raw** data.
- *"Runtime budget is not specified"* → budget is an explicit, user-visible parameter with a cost estimate.
- *"New benchmarks designed for AutoML are required"* → your dataset registry plus experiment protocol.

---

## 4. Requirements before you start

**Hardware.** Your notes record that AMLB used an AWS m5.2xlarge (8 vCPU, 32 GB). You do not need that, but plan for **8 cores and 16 GB RAM as a practical minimum** (H2O and AutoGluon are memory-hungry), and 30 to 40 GB free disk (AutoGluon pulls in PyTorch).

**Software**
- `uv` (fast, per-framework virtual environments) or conda
- Java JDK 11 or 17 (for H2O)
- Linux, WSL2 Ubuntu, or Docker (for Auto-sklearn); a C++ compiler and SWIG on that Linux side
- Python versions available side by side (uv can install 3.9, 3.10, 3.11, 3.12 on demand)
- Docker Desktop (optional but recommended for the final reproducible setup)

**Decisions to freeze in `docs/protocol.md` (my recommendation):**

| Item | Value |
|---|---|
| Split | One stratified 80/20 hold-out for the app; 3 outer folds for research experiments |
| Time budgets | Quick 60 s, Standard 300 s, Extended 900 s per framework |
| Hardware per framework | Same `n_jobs` / cores (default 4) and same memory limit for all; frameworks run **one at a time** so timings are not polluted |
| Seed | 42 everywhere; 3 seeds on a few datasets for variance |
| Input to AutoML | **Raw** data (missing values, categoricals untouched), no SMOTE. AutoML is supposed to handle preprocessing |
| Input to classic tier | Your existing leak-free preprocessing pipeline |
| Scoring | Computed centrally by acbs from predictions (never trust framework-reported scores) |
| Optimisation metric | `f1_macro` where supported, otherwise the framework's closest option (log-loss / AUC); recorded per run |
| Configuration | Vendor defaults and recommended presets, like AMLB ("near intended usage") |
| Extra baseline | `DummyClassifier` (majority class) to normalise scores |
| Statuses | `ok`, `failed`, `timeout`, `skipped`, `unavailable` (not installed), `unsupported` |

---

## 5. What changes in your measurements

| Criterion | Classic tier | AutoML tier |
|---|---|---|
| Accuracy, Precision, Recall, F1, ROC-AUC | As now | Computed from the framework's predictions by acbs |
| Training time | Wall-clock fit | Wall-clock **and** `budget_used_ratio` (used / allowed). Rank AutoML on *quality at a given budget* |
| Prediction time | ms per 1,000 rows | Same, measured on the final ensemble (often much slower: ensembles/stacking) |
| Interpretability | Score from ModelSpec | Fixed low score per framework (stacked ensembles 1; single-pipeline frameworks 2) |
| New | | `n_models_evaluated`, model summary, peak memory, framework version |

New headline question your recommender can answer: **"Is AutoML worth it here?"**, i.e. gain in F1 of best AutoML over best classic model, versus the extra training time and inference cost.

---

## 6. Step-by-step plan with IDE-agent prompts

Paste each prompt into Antigravity (or any IDE agent) one at a time. After each one: run `pytest`, `ruff check`, commit.
Start every session with: *"Read AGENTS.md and the modules named in the prompt before changing anything."*

### Step A: Housekeeping (Week 1, 1 to 2 days)

**Prompt A1**
```
Read AGENTS.md. Do housekeeping, one commit per item:
1. Run pytest and ruff. If green, commit the existing uncommitted changes in
   src/acbs/benchmark/metrics.py (_safe_roc_auc_scorer) and src/acbs/cli.py
   (sensitivity_analysis wired into generate_explanation). Add a test for
   _safe_roc_auc_scorer covering a degenerate fold (a class missing in y_true) for binary
   and multiclass.
2. Replace datetime.datetime.utcnow() with datetime.datetime.now(datetime.UTC) in
   src/acbs/api/routes/experiments.py and src/acbs/storage/models.py (use a shared helper
   utils/time.py).
3. Fix the scikit-learn deprecation warnings: LogisticRegression n_jobs usage and any SVC
   kwargs flagged by pytest -W error::FutureWarning.
4. Generate the initial Alembic migration for the current schema into alembic/versions/ and
   make the app run migrations on startup instead of Base.metadata.create_all.
Do not add features. Show me the final pytest and ruff output.
```
**Done when:** tests green, no deprecation warnings, `alembic upgrade head` works on a fresh DB.

### Step B: Compatibility spike (Week 1, 2 to 3 days). Do this BEFORE writing adapters

**Prompt B1**
```
Goal: find out which of the 9 AutoML frameworks can actually run on this machine, without
touching the main environment.

Create envs/frameworks.yaml with one entry per adapter id: autogluon, autosklearn,
autosklearn2, flaml, gama, h2o, lightautoml, mljar, naiveautoml, tpot. For each: pip
packages (pinned to a version you choose and justify), preferred python_version, system
requirements (e.g. Java for h2o; Linux + C++ compiler + swig for autosklearn), and a
smoke-test import path. NOTE: autosklearn2 uses the SAME pip package as autosklearn
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
**Done when:** `docs/framework_compat.md` exists. Frameworks that pass become "v1 supported". The rest go in a "known limitations" table (this is a result, not a failure).

### Step C: AutoML adapter layer (Weeks 2 to 3)

**Prompt C1: contract, worker and isolated runner**
```
Read src/acbs/benchmark/engine.py, metrics.py, timing.py and docs/framework_compat.md first.

Design and implement the AutoML adapter layer, keeping frameworks isolated from the main app.

1. src/acbs/automl/contracts.py (pydantic):
   AutoMLTask: dataset_id, train_path, test_path (parquet), target, task_type, time_budget_s,
     n_jobs, memory_limit_gb, seed, metric, input_mode ("raw"), framework_options (dict)
   AutoMLOutcome: status ("ok"|"failed"|"timeout"|"unavailable"|"unsupported"), error,
     framework, framework_version, fit_time_s, budget_s, budget_used_ratio,
     predict_time_per_1k_s, n_models_evaluated, model_summary (str), leaderboard (list of
     dicts, optional), peak_mem_mb, classes (ordered list), predictions_path (parquet with
     y_pred and one proba column per class, in `classes` order).
2. A tiny separate package worker/acbs_worker/ (depends only on numpy, pandas, pyarrow)
   with a CLI: python -m acbs_worker --framework <id> --task task.json --out out_dir.
   It loads train/test, calls adapters/<id>.py (class Adapter with fit(train_df, target,
   task), predict_proba(df), describe()), measures fit time with perf_counter, measures
   prediction time as the median of 3 repeats per 1,000 rows after one warm-up, writes
   predictions parquet + outcome.json. Any exception -> status "failed" with error text.
3. src/acbs/automl/runner.py: run_framework(framework_id, task) -> AutoMLOutcome. It spawns
   the worker with .envs/<id>/bin/python (Scripts\python.exe on Windows) as a subprocess,
   enforces a hard timeout = budget + grace (grace = max(25% of budget, 60 s), configurable),
   kills the whole process group on timeout, captures the last 200 lines of stdout/stderr,
   tracks peak RSS with psutil, and sets OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS to
   n_jobs. If the env does not exist -> status "unavailable" with the setup command to run.
4. src/acbs/automl/evaluate.py: given an outcome and y_test, compute ALL metrics with the
   existing compute_metrics (central scoring, never the framework's own score), verifying
   that proba columns align with `classes`.
5. A FakeAdapter used only in tests: modes ok / sleep-beyond-budget / crash / bad-class-order.
Tests: ok path, timeout is killed and reported, crash is isolated (main process survives),
missing env gives "unavailable", class-order mismatch is detected. No real framework needed.
```
**Done when:** the fake-adapter tests pass and a timeout really kills the child process.

**Prompt C2: adapters, group 1 (easiest: pure Python, sklearn-like)**
```
Implement worker/acbs_worker/adapters/ for flaml, mljar, gama and tpot. Read each
framework's current docs for exact argument names in the version pinned in
envs/frameworks.yaml before writing code (APIs change between versions; do not guess).
Mapping to honour for every adapter: time budget (seconds), n_jobs, seed, optimisation
metric (f1_macro if supported, else closest and record which), raw input with missing values
and categoricals handled the way the framework expects (document any conversion you must do,
e.g. cast object columns to category).
- flaml: AutoML().fit(task="classification", time_budget, metric, seed, n_jobs); describe()
  returns best_estimator and best_config; leaderboard from search history if available.
- mljar: AutoML(mode="Perform", total_time_limit, eval_metric, random_state, n_jobs,
  results_path in a temp dir); leaderboard from its results.
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
  presets, num_cpus). Use presets="medium_quality" for the Quick tier and "best_quality"
  only for Extended. Put its model path in a temp dir. leaderboard = predictor.leaderboard.
- h2o: h2o.init(nthreads=n_jobs, max_mem_size from memory_limit_gb); H2OAutoML(
  max_runtime_secs, seed, sort_metric, balance_classes=False). Convert the pandas frame to
  H2OFrame with the target as factor. Always shut the H2O JVM down in a finally block (a
  leaked JVM will pollute the next framework's timing and memory). Check Java is present and
  return status "unavailable" with a clear message if not.
- lightautoml: TabularAutoML(task=Task("binary" or "multiclass"), timeout, cpu_limit,
  general_params / reader_params as needed); handle the class-probability layout carefully
  (binary returns one column).
- naiveautoml: this framework has no implicit time limit and may stop early. Pass the
  version-appropriate overall timeout argument and candidate timeout (argument names changed
  between versions; read the pinned version's README), and record the real time used in
  budget_used_ratio. Do not pad the run to fill the budget.
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
sklearn computation on the saved predictions; (5) running twice with the same seed gives
the same predictions OR the adapter declares itself non-deterministic in a flag that gets
stored. Output a conformance report to docs/adapter_conformance.md.
```
**Done when:** the conformance report shows every installed framework passing, and failures are listed with reasons.

### Step D: Engine integration (Week 3)

**Prompt D1**
```
Read src/acbs/benchmark/engine.py, aggregate.py and the BenchmarkConfig.

Extend the benchmark engine so one experiment runs two tiers on the SAME hold-out split and
seed:
- Tier 1 "classic": the existing 8 models plus a DummyClassifier(strategy="most_frequent")
  baseline (store its scores; use them for normalisation in reports).
- Tier 2 "automl": the selected AutoML frameworks via src/acbs/automl/runner.py.
Add to BenchmarkConfig: automl_frameworks (list, default empty = classic only),
automl_budget_s, automl_n_jobs, automl_grace_s, automl_memory_gb, outer_folds (1 = single
hold-out, 3 = repeated for research mode).
Rules:
- Run AutoML frameworks sequentially, never in parallel, so timings are comparable.
- Write the raw train/test parquet files once per experiment; AutoML gets RAW features
  (no acbs preprocessing, no SMOTE); classic tier keeps the leak-free pipeline.
- Every result row gets: kind ("classic"|"automl"), framework, framework_version,
  budget_s, budget_used_ratio, n_models_evaluated, model_summary, peak_mem_mb, status.
- A failed/timeout/unavailable framework never stops the run (extend the existing status
  handling).
- progress_callback must report framework name and phase.
- Add estimate_cost(config, n_datasets=1) returning expected wall-clock seconds
  (sum of budgets + grace + classic time estimate) so the UI/CLI can show it before running.
CLI: acbs run --dataset adult --target income --automl flaml,mljar --budget 60.
Tests: use the FakeAdapter to verify that both tiers appear in the result, identical test
rows, and that a crashed framework yields status "failed" while all other rows are ok.
```

### Step E: Recommendation upgrade (Week 4)

**Prompt E1**
```
Read src/acbs/recommend/scoring.py and explain.py.

Upgrade the recommender to handle both tiers.
1. Candidates are now classic models and AutoML frameworks. Add interpretability scores for
   AutoML (stacked-ensemble frameworks = 1, single-pipeline frameworks = 2; put the values
   in configs/default.yaml).
2. For AutoML rows, "training time" criterion uses fit_time_s; also expose budget_used_ratio
   as an optional cost criterion. Keep log-scaling for time.
3. New function automl_value_analysis(results_df): returns best_classic, best_automl,
   delta_f1, delta_roc_auc, extra_training_time_ratio, extra_inference_time_ratio and a
   verdict string: "AutoML worth it" / "Marginal" / "Not worth it" using configurable
   thresholds (default: worth it if delta_f1 >= 0.02 and extra time <= 20x).
4. Statistical tie handling: if the top AutoML and top classic model are within the
   fold-to-fold noise (use fold scores when outer_folds > 1; otherwise use a bootstrap of the
   hold-out predictions to get a confidence interval), say so explicitly.
5. Extend generate_explanation with templates for: why this framework (strategy in one
   line, from a lookup table configs/frameworks_info.yaml containing the search strategies
   from our literature notes: AutoGluon = stacked ensembles of preset pipelines, Auto-sklearn
   = Bayesian optimisation of sklearn pipelines, Auto-sklearn 2 = Bayesian optimisation of
   iterative algorithms, FLAML = cost-frugal optimisation, GAMA = evolutionary optimisation
   of sklearn pipelines, H2O = random search + ensembling, LightAutoML = Bayesian
   optimisation of linear models and GBMs, MLJAR and NaiveAutoML = custom data-science
   pipelines, TPOT = evolutionary optimisation), the AutoML-vs-classic verdict, caveats
   (budget used, failures, non-determinism, inference latency).
6. Keep presets; add "fast_inference" behaviour that penalises ensembles automatically.
7. Sensitivity analysis and Pareto front must work across both tiers; mark AutoML points
   differently in data returned for charts (kind column).
Tests: a crafted table where AutoML wins by 5 points -> "worth it"; wins by 0.5 point at 50x
time -> "Not worth it"; weights favouring speed pick a classic model.
```

### Step F: Storage and API (Week 4)

**Prompt F1**
```
Read src/acbs/storage/ and src/acbs/api/.
1. New Alembic migration (do not edit the old one): add to model_runs the columns kind,
   framework, framework_version, budget_s, budget_used_s, n_cores, peak_mem_mb,
   model_summary, leaderboard_json, outcome_log_tail; add to experiments: hardware_json (cpu
   count, RAM, OS, python version) and outer_fold. Do not store framework install status in
   the DB; compute it live.
2. API: GET /frameworks (id, name, strategy, installed?, version, os_supported, reason if
   unavailable, default budget); POST /experiments accepts automl: {frameworks, budget_s,
   n_jobs}; GET /experiments/{id}/status reports per-framework state
   (queued/running/ok/failed/timeout/unavailable) and elapsed time; POST /estimate returns
   the cost estimate for a config; /rerank keeps working without retraining on both tiers.
3. Jobs: AutoML experiments can run for an hour. Run the experiment in a separate process
   with persisted progress so an API restart marks the experiment "interrupted" instead of
   losing it; add DELETE/cancel that kills the running framework subprocess.
Update API tests (use the FakeAdapter via a config flag so tests do not need real
frameworks).
```

### Step G: Streamlit UI with Plotly charts (Week 5)

This also finishes your "Interactive Charts Missing" gap from Antigravity.

**Prompt G1**
```
Read app/streamlit_app.py and src/acbs/reporting/charts.py. Refactor into app/pages/ and
app/components/ (experiment_form.py, framework_picker.py, charts_view.py, metrics_table.py,
recommendation_card.py), keeping UI free of ML logic and talking only to the API.

New Benchmark page: dataset profile card; target; classic models multiselect; a framework
picker listing each AutoML framework with an installed/not-installed/unsupported badge and
the reason, its one-line strategy, and a checkbox; budget selector (Quick 60 s / Standard
300 s / Extended 900 s / custom); n_jobs; and a live cost estimate ("about 47 minutes").
Block runs whose estimate exceeds a configurable limit unless the user confirms.

Progress page: per-framework status table with elapsed vs budget bars.

Results page, with the recommendation card on top (winner, explanation, stability %,
"AutoML worth it?" verdict) and tabs:
 - Overview: grouped metric bars + Pareto scatter (accuracy vs training time) with different
   markers for classic vs AutoML
 - Radar: top candidates across metrics
 - Time & cost: training time, budget used, inference latency (log scale), peak memory
 - Diagnostics: confusion matrix + ROC for the winner and a selectable model
 - Sensitivity: tornado / rank-shift chart under weight changes
 - Framework internals: each framework's leaderboard and model_summary
Weight sliders call /rerank and refresh without retraining.
Add a manual UI test checklist in docs/ui_checklist.md.
```

### Step H: Experiments for the paper (Weeks 5 to 6)

**Prompt H1**
```
Create scripts/run_automl_experiments.py and notebooks/automl_analysis.ipynb.

Datasets: all in data/registry.yaml; make sure at least 3 have missing values and at least
2 have high-cardinality categoricals, and at least 2 are strongly imbalanced (add datasets
if needed). Protocol: docs/protocol.md (Standard 300 s budget, 4 cores, seed 42, 3 outer
folds on a subset of datasets; Quick 60 s budget on all). Script must be resumable, log
every failure with its reason, store everything in the DB and results/*.csv, and print a
runtime estimate before starting.

Notebook outputs (300-dpi figures into results/figures/):
 1. Average rank of every framework and classic model across datasets, with a critical-
    difference style plot (Friedman + Nemenyi) and win/tie/loss table vs the best classic.
 2. Accuracy/F1 vs total time scatter per dataset (Pareto fronts).
 3. Failure and timeout table per framework (like AMLB's failure analysis), plus budget
    adherence (budget_used_ratio) and peak memory.
 4. Inference latency per 1,000 rows per framework (log scale).
 5. "Is AutoML worth it?" summary: delta F1 over best classic model per dataset, and the
    extra time cost.
 6. How the recommended option changes with user priorities (balanced / max_accuracy /
    fast_inference / interpretable) - this supports our novelty claim.
 7. Effect of budget: Quick vs Standard vs Extended on 3 datasets.
 8. Imbalanced datasets: accuracy vs macro-F1/recall divergence.
Also write docs/limitations.md: hardware, versions, budget-driven rankings, platform
limits, frameworks that could not run and why.
```

### Step I: Model export (optional, Week 6)

**Prompt I1**
```
Add model export for the recommended option. For classic models: refit the full pipeline on
all data and save a joblib bundle with the preprocessor, model, class labels, library
versions and a predict() usage snippet. For AutoML frameworks: ask the adapter to save the
framework's native artifact (AutoGluon predictor folder, H2O MOJO, MLJAR results folder,
pickled sklearn-style pipeline for FLAML/GAMA/TPOT where supported) into a zip with a
README that states which environment/version is needed to load it. Expose
GET /experiments/{id}/export and a Download button. Be explicit in the UI that AutoML
artifacts need the same framework version to load.
```

### Step J: Docker, docs and review material (Week 7)

**Prompt J1**
```
1. Add Docker support: a base image for the app (API + Streamlit + PostgreSQL via
   docker-compose) and one Dockerfile per framework environment under docker/frameworks/<id>/
   (Auto-sklearn on a Linux base, H2O with a JDK installed). The runner must support a
   "docker" execution backend in addition to the "venv" backend, selected in config.
2. Update README.md and docs/architecture.md with the two-tier architecture diagram
   (mermaid), the adapter contract, the protocol, how to add a new framework (one adapter
   file + one entry in envs/frameworks.yaml), and known limitations.
3. Write docs/reproducibility.md: seeds, versions, hardware recorded per run, exact commands
   to reproduce each figure in the paper.
```

---

## 7. Timeline (7 weeks from Monday 26 Oct 2026; stretch or compress to your deadline)

| Week | Steps | Output |
|---|---|---|
| 1 (26 Oct) | A, B | Clean repo, compatibility matrix, list of v1-supported frameworks |
| 2 (2 Nov) | C1, C2 | Adapter contract, isolated runner, first 4 frameworks working |
| 3 (9 Nov) | C3, C4, D1 | All installable frameworks, conformance report, engine runs both tiers from the CLI |
| 4 (16 Nov) | E1, F1 | AutoML-aware recommender, migration, API |
| 5 (23 Nov) | G1, start H1 | UI with Plotly charts and framework picker; experiments begin |
| 6 (30 Nov) | H1, I1 | Full experiments, figures, optional export |
| 7 (7 Dec) | J1 + paper | Docker, docs, paper results, demo |

**Safety milestone (end of week 3):** `acbs run --dataset adult --target income --automl flaml,mljar,autogluon --budget 60` prints one ranked table containing classic models and AutoML frameworks. If that works, everything after it is polish.

**Start the experiments (H1) as early as week 4 in the background.** They take hours, and they are what your paper needs.

---

## 8. Experiment cost estimator

Total time is roughly `datasets x frameworks x budget x folds x seeds`, because AutoML frameworks run one at a time.

| Scenario | Frameworks | Budget | Datasets | Folds/seeds | Approx. wall time |
|---|---|---|---|---|---|
| Quick check, all data | 9 | 60 s | 10 | 1 | ~1.5 h |
| Main experiment | 9 | 300 s | 10 | 1 | ~7.5 h |
| Variance study | 9 | 300 s | 3 | 3 | ~6.8 h |
| Budget study | 9 | 60/300/900 s | 3 | 1 | ~10 h |

(Add about 25% for grace periods, classic models and failures.) Run them overnight and keep the machine otherwise idle so timings are clean.

---

## 9. What to say in your next review

- **Problem:** choosing among AutoML frameworks is hard; benchmarks like AMLB compare them but do not recommend.
- **Contribution:** acbs benchmarks 9 AutoML frameworks plus classic baselines under one reproducible protocol and **recommends** the best option for the user's priorities, including whether AutoML is worth it at all.
- **Method:** isolated per-framework environments, central scoring, fixed budget/hardware, failure tracking.
- **Honest limitation:** results are specific to the budget, hardware, and framework versions used, and some frameworks may not run on every platform. This is the same limitation AMLB reports.

---

## 10. Working tips for the IDE agent

1. One prompt = one step. Do not merge steps B and C: the compatibility spike decides what is feasible.
2. Tell the agent to read each framework's documentation for the pinned version before writing its adapter. Several APIs (TPOT, NaiveAutoML, LightAutoML) have changed between releases.
3. When a framework fails to install, ask the agent to record the exact error and move on; documenting a failure is a result.
4. After every step: *"Review what you just wrote for data leakage, unhandled exceptions, hardcoded values, and missing tests. List issues, then fix them."*
5. Never let the agent install an AutoML framework into the main app environment.
