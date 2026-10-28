/* pages/results.js — recommendation, charts, diagnostics, rerank, advisor. */

const { apiGet, apiPost, qp } = ACBS;
const {
  esc,
  badge,
  tableHtml,
  emptyState,
  fmtSec,
  fmtNum,
  fmtTime,
} = window.ui;

const id = qp("id");
let RESULT = null;
let REC = null;

const WEIGHTS = [
  ["accuracy", "Accuracy"],
  ["f1", "F1"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["roc_auc", "ROC AUC"],
  ["train_time", "Train speed"],
  ["predict_time", "Predict speed"],
  ["interpretability", "Interpretability"],
];

/* ---------- loading ---------- */

async function loadAll() {
  const body = document.getElementById("body");
  if (!id) {
    body.innerHTML = emptyState(
      "No experiment selected",
      "Open one from History, or start a new benchmark.",
      '<a class="btn primary mt-4" href="new-benchmark.html">New benchmark</a>'
    );
    return;
  }
  try {
    RESULT = await apiGet(`/experiments/${id}/results`);
  } catch (e) {
    body.innerHTML = emptyState(
      "Results not ready",
      e.message,
      `<a class="btn mt-4" href="run-monitor.html?id=${id}">Open run monitor</a>`
    );
    return;
  }
  REC = await apiGet(`/experiments/${id}/recommendation`).catch(() => null);
  body.innerHTML = "";
  document.getElementById("content").style.display = "";
  document.getElementById("page-title").textContent = `Results · experiment #${id}`;
  const cfg = RESULT.config || {};
  document.getElementById("page-sub").textContent =
    `${(RESULT.profile && RESULT.profile.n_rows) || "?"} rows · ` +
    `${cfg.cv_folds || "?"} folds · seed ${RESULT.seed} · primary metric ${cfg.metrics_primary || "—"}`;
  wireExports();
  renderRec();
  renderOverview();
  renderRuns();
  renderDiag();
  renderFailures();
  renderRerank();
  wireAdvisor();
  shell.wireTabs();
}

/* ---------- recommendation ---------- */

function renderRec() {
  const box = document.getElementById("rec");
  if (!REC) {
    box.innerHTML = `<div class="notice">Recommendation unavailable for this experiment.</div>`;
    return;
  }
  const ex = REC.explanation || {};
  const paras = (ex.paragraphs || []).map((p) => `<li>${esc(p)}</li>`).join("");
  const whyNot = (ex.why_not || []).map((p) => `<li><span class="mono small">${esc(p)}</span></li>`).join("");
  const caveats = (ex.caveats || []).map((p) => `<li>${esc(p)}</li>`).join("");
  box.innerHTML = `
    <div class="rec-card">
      <div class="row between">
        <div>
          <div class="small faint">${esc(ex.headline || "Recommended")}</div>
          <div class="model">${esc(REC.model_name)}</div>
        </div>
        <div class="row">
          <div class="stat" style="padding: 10px 16px">
            <div class="label">Composite</div>
            <div class="value" style="font-size: 18px">${fmtNum(ex.score, 3)}</div>
          </div>
          <div class="stat" style="padding: 10px 16px">
            <div class="label">Stability</div>
            <div class="value" style="font-size: 18px">${fmtNum(REC.stability_pct, 0)}%</div>
          </div>
        </div>
      </div>
      <ul>${paras}</ul>
      ${whyNot ? `<div class="small faint">Why not the others</div><ul class="mono small">${whyNot}</ul>` : ""}
      ${caveats ? `<div class="small faint">Caveats</div><ul>${caveats}</ul>` : ""}
    </div>`;
}

/* ---------- overview ---------- */

function renderOverview() {
  const table = REC && Array.isArray(REC.score_table) ? REC.score_table : [];
  if (!table.length) {
    document.getElementById("score-table").innerHTML = emptyState("No scored rows", "—");
    return;
  }
  const recName = REC.model_name;
  const metric = (RESULT.config && RESULT.config.metrics_primary) || "f1_macro";

  // bars: composite score
  document.getElementById("bars-title").textContent = "Composite score";
  window.charts.chartMetricBars(
    "bars",
    table.map((r) => ({
      model_name: r.model_name,
      value: r.composite_score,
      excluded: r.excluded,
      is_rec: r.model_name === recName,
    })),
    "Composite score"
  );

  // pareto needs training time + metric
  const hasMetric = table.some((r) => r[metric] != null && r.training_time != null);
  if (hasMetric) {
    window.charts.chartPareto(
      "pareto",
      table.map((r) => ({ ...r, model_name: r.model_name })),
      metric
    );
  } else {
    document.getElementById("pareto").innerHTML =
      '<p class="faint small">Not enough numeric columns for a Pareto plot.</p>';
  }

  // dynamic score table
  const preferred = ["rank", "model_name", "kind", "framework", "composite_score", metric,
    "accuracy", "roc_auc", "training_time", "prediction_time_per_1k", "budget_used_ratio"];
  const keys = [];
  preferred.forEach((k) => {
    if (table.some((r) => r[k] !== undefined && r[k] !== null)) keys.push(k);
  });
  Object.keys(table[0]).forEach((k) => {
    if (!keys.includes(k) && !["best_params", "y_true", "y_pred", "y_proba", "fold_scores",
      "cv_metrics", "test_metrics", "model_summary"].includes(k)) keys.push(k);
  });
  const cols = keys.map((k) => ({
    key: k,
    label: k.replace(/_/g, " "),
    cls: typeof table[0][k] === "number" ? "num" : "",
    fmt: (r) => {
      const v = r[k];
      if (v === null || v === undefined) return '<span class="faint">—</span>';
      if (typeof v === "number") return esc(Number.isInteger(v) ? v : v.toFixed(3));
      if (typeof v === "boolean") return v ? "yes" : "no";
      return esc(v);
    },
  }));
  document.getElementById("score-table").innerHTML = tableHtml(cols, table, {
    empty: emptyState("Empty score table", "No scored models."),
  });
}

/* ---------- runs ---------- */

function renderRuns() {
  const rows = (RESULT.models || []).map((m) => ({
    ...m,
    _cls: m.model_name === (REC && REC.model_name) ? "" : "",
  }));
  const cols = [
    {
      key: "model_name",
      label: "Model",
      fmt: (r) =>
        `<a href="run-detail.html?id=${id}&model=${encodeURIComponent(r.model_name)}">${esc(r.model_name)}</a>`,
    },
    { key: "kind", label: "Kind", fmt: (r) => `<span class="faint">${esc(r.kind)}</span>` },
    { key: "status", label: "Status", fmt: (r) => badge(r.status) },
    {
      key: "primary",
      label: "Primary",
      cls: "num",
      fmt: (r) => {
        const metric = (RESULT.config && RESULT.config.metrics_primary) || "f1_macro";
        const v = (r.test_metrics || {})[metric] ?? (r.cv_metrics || {})[metric];
        return v != null ? esc(Number(v).toFixed(4)) : '<span class="faint">—</span>';
      },
    },
    {
      key: "roc_auc",
      label: "ROC AUC",
      cls: "num",
      fmt: (r) => {
        const v = (r.test_metrics || {}).roc_auc ?? (r.cv_metrics || {}).roc_auc;
        return v != null ? esc(Number(v).toFixed(4)) : '<span class="faint">—</span>';
      },
    },
    {
      key: "training_time",
      label: "Train",
      cls: "num",
      fmt: (r) => esc(fmtSec(r.training_time)),
    },
    {
      key: "prediction_time_per_1k",
      label: "Predict/1k",
      cls: "num",
      fmt: (r) => (r.prediction_time_per_1k != null ? `${fmtNum(r.prediction_time_per_1k * 1000, 1)} ms` : "—"),
    },
    {
      key: "budget_used_ratio",
      label: "Budget",
      cls: "num",
      fmt: (r) =>
        r.budget_used_ratio != null ? `${fmtNum(r.budget_used_ratio * 100, 0)}%` : '<span class="faint">—</span>',
    },
  ];
  document.getElementById("runs-table").innerHTML = tableHtml(cols, rows);
}

/* ---------- diagnostics ---------- */

function renderDiag() {
  const models = RESULT.models || [];
  const pick =
    models.find((m) => REC && m.model_name === REC.model_name && m.y_true && m.y_true.length) ||
    models.find((m) => m.y_true && m.y_true.length);
  if (!pick) {
    document.getElementById("cm").innerHTML =
      '<p class="faint small">No stored predictions for this experiment.</p>';
    document.getElementById("roc").innerHTML = "";
    document.getElementById("calib").innerHTML = "";
    return;
  }
  const classes = RESULT.classes && RESULT.classes.length ? RESULT.classes : [...new Set([...pick.y_true, ...pick.y_pred])];
  document.getElementById("cm-sub").textContent = `${pick.model_name}, hold-out split.`;
  if (pick.y_pred && pick.y_pred.length) {
    window.charts.chartConfusion("cm", pick.y_true, pick.y_pred, classes);
  }
  if (pick.y_proba && pick.y_proba.length) {
    window.charts.chartRoc("roc", pick.y_true, pick.y_proba, classes);
    window.charts.chartCalibration("calib", pick.y_true, pick.y_proba, classes);
  } else {
    document.getElementById("roc").innerHTML =
      '<p class="faint small">No probabilities stored (model does not expose predict_proba).</p>';
    document.getElementById("calib").innerHTML = "";
  }
}

/* ---------- failures ---------- */

function renderFailures() {
  const bad = (RESULT.models || []).filter((m) => m.status !== "ok");
  const box = document.getElementById("fail-table");
  if (!bad.length) {
    box.innerHTML = `<div class="notice">Every run completed — no failures or timeouts recorded.</div>`;
    return;
  }
  const cols = [
    { key: "model_name", label: "Model" },
    { key: "status", label: "Status", fmt: (r) => badge(r.status) },
    {
      key: "error",
      label: "Recorded error",
      fmt: (r) => `<span class="mono small">${esc((r.error || "").split("\n")[0].slice(0, 160))}</span>`,
    },
  ];
  box.innerHTML = tableHtml(cols, bad);
}

/* ---------- rerank ---------- */

function renderRerank() {
  const box = document.getElementById("weights");
  const defaults = { accuracy: 0.15, f1: 0.25, precision: 0.1, recall: 0.15, roc_auc: 0.15, train_time: 0.1, predict_time: 0.05, interpretability: 0.05 };
  box.innerHTML = WEIGHTS.map(
    ([key, label]) => `
      <div class="range-row">
        <span>${label}</span>
        <input type="range" min="0" max="1" step="0.05" value="${defaults[key]}" data-w="${key}" />
        <span class="val" data-v="${key}">${defaults[key].toFixed(2)}</span>
      </div>`
  ).join("");
  box.querySelectorAll("input[type=range]").forEach((r) =>
    r.addEventListener("input", () => {
      box.querySelector(`[data-v="${r.dataset.w}"]`).textContent = (+r.value).toFixed(2);
    })
  );

  document.getElementById("rerank-btn").addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    const prefs = {};
    box.querySelectorAll("input[type=range]").forEach((r) => (prefs[r.dataset.w] = +r.value));
    prefs.method = document.getElementById("method").value;
    try {
      REC = await apiPost(`/experiments/${id}/rerank`, { preferences: prefs });
      toast("Re-ranked", "ok");
      renderRec();
      renderOverview();
      renderRuns();
    } catch (e) {
      toast(e.message, "error");
    } finally {
      btn.disabled = false;
    }
  });
}

/* ---------- advisor ---------- */

function wireAdvisor() {
  const ask = async (question) => {
    const out = document.getElementById("advisor-out");
    out.innerHTML = '<p class="faint small">Asking the advisor…</p>';
    try {
      const res = await apiPost(`/experiments/${id}/advisor`, { question: question || null });
      out.innerHTML = `<div class="notice">${esc(res.narrative)}</div>`;
    } catch (e) {
      out.innerHTML = `<div class="notice warn">${esc(e.message)} — showing the stored explanation instead.</div>`;
    }
  };
  document.getElementById("advisor-btn").addEventListener("click", () =>
    ask(document.getElementById("advisor-q").value.trim())
  );
  document.getElementById("advisor-q").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") ask(ev.target.value.trim());
  });
  document.querySelectorAll("#chips .chip").forEach((c) =>
    c.addEventListener("click", () => ask(c.dataset.q))
  );
}

/* ---------- exports ---------- */

function wireExports() {
  const base = `/experiments/${id}`;
  document.getElementById("exports").style.display = "";
  document.getElementById("btn-html").href = `${base}/report?format=html`;
  document.getElementById("btn-csv").href = `${base}/report?format=csv`;
  document.getElementById("btn-pdf").href = `${base}/report?format=pdf`;
  document.getElementById("btn-zip").href = `${base}/export-model`;
}

document.addEventListener("DOMContentLoaded", () => {
  loadAll().catch((e) => {
    document.getElementById("body").innerHTML = emptyState("Failed to load", e.message);
  });
});
