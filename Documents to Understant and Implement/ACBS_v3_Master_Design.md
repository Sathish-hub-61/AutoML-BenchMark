# ACBS v3: Master Design (AutoML-only, AMLB-informed, with an AI Advisor Agent)

Auto Classifier Benchmarking System (package `acbs`). This document is the single source of truth for v3. Every other file (prompts, pages blueprint, integration plan, tracker) follows it. If they disagree, this file wins.

## 1. What changed from v2 and why

| Area | v2 | v3 | Reason |
|---|---|---|---|
| Contenders | 8 classic models + 9 AutoML frameworks (two tiers) | **9 AutoML frameworks only**, plus one non-model reference (constant predictor) | Mentor/AutoML scope; the classic models were "basic ML". |
| Tuning, preprocessing, SMOTE | Own Optuna/RandomizedSearchCV, own pipelines | **Removed.** Each framework gets raw data and does its own preprocessing and search | AMLB principle: frameworks run with default settings; only mode, time, cores, memory and metric are controlled. |
| Frontend | Streamlit | **Plain HTML, CSS, JavaScript** (9 pages, no build step), FastAPI under `/api` | Requested. |
| Methodology | Own design | **AMLB-style protocol** (Section 3) | Defensible, published, comparable. |
| AI | Template explanations only | **Benchmark Advisor Agent** (tool-calling LLM) with template fallback | Mentor requirement. |
| "AutoML worth it?" | Compared AutoML against classic models | Replaced by **gain over the constant-predictor floor** and the **budget ladder** (score versus time budget) | No classic tier left to compare against. |

Kept from v1: SQLAlchemy storage, FastAPI, ranking (min-max, TOPSIS), Pareto and sensitivity analysis, template explanations (agent fallback), HTML/PDF reporting, Typer CLI, test structure.
Retired from v1 (tag the repository `v1-classic` first): `models/` (8 classic models), `tuning/`, `preprocessing/` pipelines, imbalance handling (SMOTE), the Streamlit app, and the tests that only covered them.

## 2. What AMLB teaches us (analysis)

Sources: [AMLB site](https://openml.github.io/automlbenchmark/), [github.com/openml/automlbenchmark](https://github.com/openml/automlbenchmark), the [JMLR paper](https://jmlr.org/papers/v25/22-0493.html) (9 frameworks, 71 classification + 33 regression tasks) and the [shorter-budget study](https://arxiv.org/html/2504.01222v1).

| AMLB practice | What we adopt | Where |
|---|---|---|
| Frameworks optimise the same metric they are evaluated on: AUC (binary), log loss (multiclass) | Every run passes the chosen target metric to the framework; defaults AUC / log loss; user may choose accuracy, balanced accuracy or macro-F1 | Task contract, New Benchmark |
| Default framework settings; only mode, runtime, cores, memory, metric are controlled | `FrameworkSpec` holds a fixed "mode" per variant (for example AutoGluon `best_quality`/`high_quality`, MLJAR `Compete`/`Perform`) and no free hyperparameters | Registry, Frameworks page |
| Runtime is enforced; other constraints are passed through but not always respected | Hard wall-clock kill (budget + grace) enforced by the runner; cores and memory limits passed to the framework and enforced by Docker backend; the report states which limits are enforced | Runner |
| Extra time beyond the search budget for loading, prediction and cleanup, not told to the framework | `grace` seconds, not told to the framework | Runner |
| Failures (crash, timeout) are not random and are not dropped | Failures are stored with a reason; in aggregate scoring a failed run is imputed with the constant-predictor score, and the failure count is always shown | Scoring, Results |
| Constant predictor and random forest baselines; scores scaled so that baseline = 0 and best = 1 | **Constant predictor** is the floor (mandatory, not a contender). Scaled score = (score - floor) / (best - floor). The random-forest reference is an optional toggle, off by default, because the project no longer ships basic ML | Scoring |
| Outer cross-validation folds from the task definition; one framework run per fold | N outer stratified folds (default 3, up to 10); AutoML sees only the training part and does its own internal validation | Runner, Scoring |
| Inference time as rows per second: one row in memory and 10,000 rows, median of 10 repeats | Same protocol, measured inside the framework environment | Task contract, Results |
| Critical difference diagram (Friedman + Nemenyi), scaled-score plots, box plots, Bradley-Terry trees | Critical difference and scaled-score plots when 3 or more datasets are benchmarked; with one dataset, fold-wise ranks and a Wilcoxon signed-rank test. Bradley-Terry trees are a stretch goal | Results, Methodology |
| `results.csv` with `id, task, fold, framework, constraint, result, metric, duration, seed` and per-run prediction files | Store the same columns plus our extras; **export an AMLB-compatible results.csv** | Export, History |
| Runs in local venv, Docker, Singularity or AWS | `venv` and `docker` backends now; AWS/remote runners are future work | Runner |
| OpenML suites and tasks plus custom datasets | Dataset sources: upload CSV/Parquet, **OpenML task or suite ID**, and a small built-in suite | Datasets |
| Known limits: auto-sklearn 2 meta-learning may be optimistic on public datasets; NaiveAutoML was excluded from the AMLB analysis; early stopping changes results | These appear as caveats on the Methodology and Results pages. NaiveAutoML is labelled **experimental**. Early stopping is an optional, separately labelled configuration | Methodology |
| Findings: rankings are consistent across budgets (correlation above 0.96); a 30-minute budget tracks 60 minutes closely; 5 minutes is noisier; high-dimensional data is less stable | Budgets: Smoke 1 min, Quick 5 min, Standard 15 min, Thorough 30 min. The **budget ladder** runs selected frameworks at several budgets and plots score against time | Budgets, Results |

Honest caveat for the report: AMLB used 8 vCPU / 32 GB cloud machines and 1 to 4 hour budgets over 104 tasks. A student project on a laptop gives much smaller evidence, so results must be presented as "on this hardware, on these datasets, with these budgets".

## 3. Evaluation protocol (v3)

1. **Dataset intake**: upload, OpenML task/suite, or built-in suite. Profile it (rows, features, types, missing values, class balance). Reject unsafe or unusable data with plain-language reasons.
2. **Split**: outer stratified K folds with seed 42 (default K = 3, maximum 10), identical for every framework. Optional repeated folds are not in the first release.
3. **Per run** (framework x fold): the runner builds an `AutoMLTask` (train file, test file, target, metric, budget seconds, cores, memory, seed) and starts the framework in its own environment. The framework returns predictions (labels and probabilities), its own reported info and timing, and the inference benchmark.
4. **Scoring is central**: ACBS computes every metric from the returned predictions (never trusts framework-reported scores). Primary metric plus accuracy, balanced accuracy, macro-F1, AUC, log loss where applicable.
5. **Cost measurements**: wall-clock fit time, budget used, peak memory, inference rows/second (single-row and 10,000-row), model/artifact size where available.
6. **Failures**: statuses `ok, failed, timeout, skipped, unavailable, unsupported`. Each non-ok run keeps its reason and log tail.
7. **Aggregation**: mean and standard deviation across folds; scaled score against the constant-predictor floor; failure-imputed ranks; statistical tests (Section 2).
8. **Recommendation**: weighted multi-criteria ranking (min-max and TOPSIS) over: primary metric, fold stability (standard deviation), training time, inference throughput, memory, failure rate. Presets: Balanced, Accuracy first, Fast inference, Low resource, Reliable. `/rerank` changes weights without retraining.
9. **Fairness rules**: frameworks run sequentially (never in parallel) on the same machine with the same cores, memory, seed, folds and metric; versions, operating system and hardware are recorded with every run.
10. **Reproducibility statement**: time-bounded search is not bit-for-bit repeatable. We store seed, versions, constraints and predictions so that scores can be recomputed, and say so in the report.

## 4. Architecture (v3)

```
Browser (HTML/CSS/JS, web/)  --->  FastAPI  (/api, serves web/)
                                       |
          +----------------------------+---------------------------+
          |                            |                           |
     Services layer             Benchmark runner            Advisor Agent
 (datasets, experiments,      (queue, sequential,         (LLM + tools that call
  scoring, ranking, reports)   venv/docker backends)         the same services)
          |                            |
      SQLAlchemy DB            Framework workers: .envs/<id> or docker/<id>
   (SQLite / PostgreSQL)       AutoGluon, Auto-sklearn (+2), FLAML, GAMA, H2O,
                               LightAutoML, MLJAR, NaiveAutoML, TPOT
```

Rules: no ML logic in the frontend or API routes; the agent can only call services through the tool layer; the worker process never touches the database.

Package layout: `src/acbs/{api,services,datasets,automl(registry,runner,contract,adapters),scoring,ranking,reporting,agent,db,cli}`, `workers/<framework>/`, `web/{*.html,css/,js/{api.js,shell.js,charts.js,components/,pages/},vendor/,content/}`, `configs/frameworks.yaml`, `docker/`.

## 5. The Benchmark Advisor Agent

**What it is**: an LLM that decides which service to call next (tool calling in a loop) and explains results in plain language. It never computes or invents metrics.

**What it can do**
- Read the dataset profile and propose a framework set, budget, metric and priority preset, each with a reason.
- Show the plan and wait for the user to approve (the approval gate is enforced by the backend, not by the prompt).
- Start the approved benchmark, watch progress, report failures in plain words.
- Explain results: why the winner won, trade-offs, stability, failures, caveats; answer follow-ups such as "what if inference speed mattered more?" by calling `/rerank`.
- Suggest a next experiment (for example the budget ladder for the top two frameworks).

**Tools** (thin wrappers over services, JSON-schema typed): `profile_dataset`, `list_frameworks`, `estimate_cost`, `propose_plan` (returns a plan awaiting approval), `start_benchmark` (only accepts an approved plan id), `get_status`, `get_results`, `rerank`, `compare_runs`, `explain_failure`.

**Guardrails**
- Approval required before any run starts; hard caps on budget, number of frameworks and tool calls per conversation.
- Grounding: every number in a final answer must exist in a tool result; a post-check rejects ungrounded numbers and falls back to the template explanation.
- Data and column names from datasets are treated as untrusted text (prompt-injection safe): they are passed as data, never as instructions, and tool calls are validated against schemas.
- No secrets in prompts or logs; API key read from environment; provider is pluggable (Claude, OpenAI, Gemini or local Ollama) behind one `LLMClient` interface.
- Every session, message and tool call is stored (`agent_sessions`, `agent_messages`, `agent_tool_calls`) and shown in a collapsible trace.
- If no LLM is configured the Advisor panel shows rule-based suggestions and the template explanation.

**UI**: a docked Advisor panel available on New Benchmark, Run Monitor and Results (not a tenth page). Suggestion chips ("Plan a benchmark for this dataset", "Explain this result", "Why did a framework fail?").

**Testing**: a `FakeLLM` that replays scripted tool calls, so the whole agent loop is tested without network access; evaluation set of 15 questions with expected tool sequences.

## 6. Pages (frontend)

Nine HTML pages, unchanged in purpose: Home, Datasets, New Benchmark, Run Monitor, Results, Model Detail (now "Run Detail": one framework run on one fold), History, Frameworks, Methodology.

| Page | v3 changes |
|---|---|
| Datasets | Add OpenML task/suite import and the built-in suite; imbalance shown as a warning, not "fixed" |
| New Benchmark | Remove the classic-model picker. Framework picker with variant (mode) selector, status badge and reason. Metric picker, folds (1 to 10), budget presets (1/5/15/30 min) or custom, cores and memory. Live cost estimate = frameworks x folds x (budget + grace). Optional toggles: budget ladder, early-stopping variant, random-forest reference. Advisor panel |
| Run Monitor | Grid of framework x fold cells with status and elapsed vs budget |
| Results | Recommendation card; tabs: Overview (scaled-score bars and Pareto of score vs training time), Rankings (average ranks, critical-difference diagram when 3+ datasets, Wilcoxon for one dataset), Time and cost, Inference (rows/second), Diagnostics (confusion matrix, ROC, calibration), Sensitivity, Budget ladder, Failures. Export: PDF, AMLB-format results.csv, predictions |
| Run Detail | Framework's own leaderboard/model summary, logs, predictions preview |
| Frameworks | Registry with version, mode variants, strategy one-liner, install state, smoke test, "how to fix" command, OS support (Auto-sklearn Linux only; H2O needs Java) |
| Methodology | The protocol of Section 3, AMLB references, caveats, limitations |

## 7. Data model additions

`datasets`, `experiments` (config JSON, metric, folds, budget, seed), `runs` (experiment, framework, variant, fold, status, reason, fit_seconds, budget_used, peak_mem_mb, infer_rows_per_s_single, infer_rows_per_s_batch, artifacts path), `predictions` (run, row index, y_true, y_pred, proba), `scores` (run, metric, value), `framework_versions`, `agent_sessions`, `agent_messages`, `agent_tool_calls`. Alembic migrations from the first table onward.

## 8. Roadmap (7 weeks, adjust to the real deadline)

| Week | Deliverable |
|---|---|
| 0 | Tag `v1-classic`; compatibility spike for all 9 frameworks on your OS/Python versions; decide Docker or WSL2 |
| 1 | Retire classic tier; contract, registry, runner (venv backend), FakeAdapter, constant-predictor floor; `/api` move |
| 2 | Adapters for AutoGluon, FLAML, H2O, MLJAR, LightAutoML (the easy five); central scoring; inference benchmark |
| 3 | Adapters for Auto-sklearn (+2), GAMA, TPOT, NaiveAutoML; failure handling; Docker backend |
| 4 | Aggregation, ranking, statistics, budget ladder; AMLB-format export; OpenML intake |
| 5 | Frontend foundation and the first 5 pages |
| 6 | Remaining pages, charts, Advisor Agent (tools, guardrails, panel) |
| 7 | Test, security review, E2E, performance, documentation, demo, report |

Cut order if time runs short: Bradley-Terry trees, early-stopping variant, Singularity/AWS, random-forest reference, Wilcoxon extras, then the budget ladder. Never cut: scoring correctness, failure isolation, fair-run rules, the agent's approval gate.

## 9. Decisions that need your approval

1. Confirm the classic tier is fully removed (only the constant-predictor floor remains as a non-contender).
2. Hardware and OS for the final demo (Auto-sklearn needs Linux; use WSL2 or Docker on Windows).
3. Default budgets 1/5/15/30 minutes and default folds = 3 (a full 9-framework run at Standard with 3 folds is about 7 hours of compute per dataset; Quick is about 2.3 hours).
4. LLM provider for the agent (API key or local Ollama) and whether the agent may launch runs after approval.
5. API under `/api`, plain JS frontend, vendored Plotly.js.
6. Whether NaiveAutoML stays in the default set even though AMLB excluded it from its analysis (recommended: keep, labelled experimental).
