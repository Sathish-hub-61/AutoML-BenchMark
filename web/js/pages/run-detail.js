/* pages/run-detail.js — one framework run: metrics, folds, summary, logs. */

const { apiGet, qp } = ACBS;
const { esc, badge, tableHtml, emptyState, fmtSec, fmtNum, fmtBytesMb } = window.ui;

const id = qp("id");
const model = qp("model");

function kv(pairs) {
  return `<dl class="kv">${pairs
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`)
    .join("")}</dl>`;
}

async function load() {
  const body = document.getElementById("body");
  if (!id || !model) {
    body.innerHTML = emptyState("Missing experiment or model", "Open a run from the results page.");
    return;
  }
  document.getElementById("back").href = `results.html?id=${id}`;
  document.getElementById("title").textContent = model;
  let result;
  try {
    result = await apiGet(`/experiments/${id}/results`);
  } catch (e) {
    body.innerHTML = emptyState("Results not ready", e.message,
      `<a class="btn mt-4" href="run-monitor.html?id=${id}">Run monitor</a>`);
    return;
  }
  const m = (result.models || []).find((x) => x.model_name === model);
  if (!m) {
    body.innerHTML = emptyState("Run not found", `No run named "${model}" in experiment #${id}.`);
    return;
  }

  const metric = (result.config && result.config.metrics_primary) || "f1_macro";
  document.getElementById("sub").textContent =
    `experiment #${id} · ${m.kind}${m.framework ? ` · ${m.framework}` : ""}` +
    (m.framework_version ? ` v${m.framework_version}` : "");

  const cvRows = Object.entries(m.cv_metrics || {}).map(([k, v]) => ({ metric: k, value: v }));
  const testRows = Object.entries(m.test_metrics || {}).map(([k, v]) => ({ metric: k, value: v }));
  const foldRows = Object.entries(m.fold_scores || {}).map(([k, vals]) => ({
    metric: k,
    scores: (vals || []).map((v) => Number(v).toFixed(4)).join(", "),
    mean: (vals && vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN),
  }));

  const metricCols = [
    { key: "metric", label: "Metric" },
    { key: "value", label: "Value", cls: "num", fmt: (r) => esc(Number(r.value).toFixed(4)) },
  ];

  body.innerHTML = `
    <div class="grid cols-2">
      <div class="card">
        <h3>Run</h3>
        <div id="kv-run"></div>
      </div>
      <div class="card">
        <h3>Cost</h3>
        <div id="kv-cost"></div>
      </div>
    </div>

    ${m.status !== "ok" ? `<div class="notice danger mt-4"><strong>${esc(m.status)}</strong> — <span class="mono small">${esc((m.error || "").split("\n")[0])}</span></div>` : ""}

    <div class="grid cols-2 mt-4">
      <div class="card">
        <h3>Hold-out metrics</h3>
        <div id="test-metrics"></div>
      </div>
      <div class="card">
        <h3>CV metrics (mean)</h3>
        <div id="cv-metrics"></div>
      </div>
    </div>

    ${foldRows.length ? `
    <div class="card mt-4">
      <h3>Fold scores</h3>
      <div id="fold-scores"></div>
    </div>` : ""}

    ${m.model_summary ? `
    <div class="card mt-4">
      <h3>Framework's own summary</h3>
      <pre class="mono small" style="white-space: pre-wrap; color: var(--text-dim)">${esc(m.model_summary)}</pre>
    </div>` : ""}

    ${m.notes ? `<div class="notice mt-4">${esc(m.notes)}</div>` : ""}
  `;

  document.getElementById("kv-run").innerHTML = kv([
    ["Status", badge(m.status)],
    ["Kind", esc(m.kind)],
    ["Framework", esc(m.framework || m.model_name)],
    ["Version", esc(m.framework_version || "—")],
    ["Primary metric", `<span class="mono">${esc(metric)}</span>`],
    ["Interpretability", m.interpretability_score != null ? `${m.interpretability_score}/5` : "—"],
  ]);

  document.getElementById("kv-cost").innerHTML = kv([
    ["Training time", esc(fmtSec(m.training_time))],
    ["Prediction / 1k rows", m.prediction_time_per_1k != null ? `${fmtNum(m.prediction_time_per_1k * 1000, 1)} ms` : "—"],
    ["Budget", m.budget_s != null ? fmtSec(m.budget_s) : "—"],
    ["Budget used", m.budget_used_ratio != null ? `${fmtNum(m.budget_used_ratio * 100, 1)}%` : "—"],
    ["Peak memory", m.peak_mem_mb != null ? fmtBytesMb(m.peak_mem_mb) : "—"],
    ["Models evaluated", m.n_models_evaluated != null ? m.n_models_evaluated : "—"],
  ]);

  document.getElementById("test-metrics").innerHTML = testRows.length
    ? tableHtml(metricCols, testRows)
    : '<p class="faint small">No hold-out metrics.</p>';
  document.getElementById("cv-metrics").innerHTML = cvRows.length
    ? tableHtml(metricCols, cvRows)
    : '<p class="faint small">No CV metrics.</p>';
  if (foldRows.length) {
    document.getElementById("fold-scores").innerHTML = tableHtml(
      [
        { key: "metric", label: "Metric" },
        { key: "scores", label: "Per fold", cls: "num" },
        { key: "mean", label: "Mean", cls: "num", fmt: (r) => esc(Number.isNaN(r.mean) ? "—" : r.mean.toFixed(4)) },
      ],
      foldRows
    );
  }
}

document.addEventListener("DOMContentLoaded", () =>
  load().catch((e) => {
    document.getElementById("body").innerHTML = emptyState("Failed to load", e.message);
  })
);
