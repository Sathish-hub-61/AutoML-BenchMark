# UI Checklist — Step G Streamlit Refactor

This checklist must be verified manually after launching the Streamlit app
(`streamlit run app/streamlit_app.py`) with the API running
(`uvicorn acbs.api.main:app --reload --port 8000`).

---

## Navigation & Layout

- [ ] Sidebar shows "API connected" (green) when the API is live
- [ ] Sidebar shows a clear error + startup command when the API is down (cached — no per-rerun stall)
- [ ] No duplicate native `pages/` navigation (page modules live in `app/views/`)
- [ ] All five nav items (New Benchmark, Progress, Results, History, Export) switch the page correctly
- [ ] Metrics guide is visible in the sidebar

---

## New Benchmark page

- [ ] Page offers **one primary action**: upload a CSV → "🚀 Upload & Run Benchmark"
- [ ] Target column is auto-detected and shown (`Target column: … (auto-detected)`)
- [ ] No framework checkboxes in the default flow — the system selects installed frameworks itself
- [ ] Upload → run redirects straight to the Progress page (no further clicks)
- [ ] "⚙️ Advanced settings" expander still exposes:
  - [ ] Built-in registry loader, classic-model multiselect, tuning/CV/trial controls
  - [ ] "Choose AutoML frameworks myself" → the 10 framework cards with correct badges
  - [ ] Budget presets (Auto / Quick / Standard / Extended / Custom) + n_jobs slider
  - [ ] Priority preset and custom weights
- [ ] Estimate > 1 hour (manual mode) → warning + confirmation checkbox appears
- [ ] "Run Benchmark" button disabled until dataset is loaded

---

## Progress page

- [ ] Experiment ID input defaults to last run experiment
- [ ] Progress bar updates percentage and message
- [ ] Auto-refreshes every 2 s via a fragment (no full-page freeze, no sleep)
- [ ] Per-framework status table shows elapsed vs budget mini-bars
- [ ] Status icons (🟢🔵⏳🔴🟠⬛) render correctly for each status
- [ ] "Cancel" button appears when status is queued/running
- [ ] Auto-started runs jump to Results automatically when finished

---

## Results page

### Recommendation Card
- [ ] Winner name and stability % displayed prominently
- [ ] AutoML verdict badge shows with correct colour:
  - "AutoML worth it" → green
  - "Marginal" → orange
  - "Not worth it" → red
- [ ] Full explanation expander shows paragraphs
- [ ] Value analysis metrics (Δ F1, training ratio, inference ratio) displayed

### Overview tab
- [ ] Grouped bar chart renders with classic (teal) vs automl (purple) colours
- [ ] Pareto scatter (accuracy vs train_time_s, log-scale x) shows all models
- [ ] Hovering shows model names

### Radar tab
- [ ] Top-N slider (3–10) controls how many models appear
- [ ] Radar chart closes correctly (last value = first value)

### Time & Cost tab
- [ ] Training time bar chart (log scale) renders
- [ ] Inference latency bar chart (log scale) renders
- [ ] Budget used ratio chart renders for AutoML rows
- [ ] Peak memory chart renders when `peak_mem_mb` data present

### Leaderboard tab
- [ ] Coloured DataFrame with RdYlGn gradient on metric columns
- [ ] `kind` column coloured purple (automl) / teal (classic)
- [ ] `status` column: non-ok values in red

### Sensitivity tab
- [ ] Horizontal bar chart of weighted scores renders
- [ ] Live rerank sidebar: preset selector + "Apply" button updates recommendation

### Framework Internals tab
- [ ] AutoML-only rows shown; each in an expander
- [ ] F1, ROC-AUC, train time, budget used, model_summary, n_models_evaluated shown

---

## History page

- [ ] Table of past experiments with coloured status cells
- [ ] "Open →" button loads the experiment ID into session state
- [ ] Switching to Results page shows that experiment's data

---

## Export page

- [ ] HTML / CSV / JSON / PDF links open/download correctly for selected experiment ID

---

## API Registration (backend)

- [ ] `GET /frameworks` returns list with installed/unsupported status
- [ ] `POST /frameworks/estimate` returns `total_wall_seconds` and breakdown
- [ ] `GET /experiments/{id}/status` includes `framework_statuses` when applicable
