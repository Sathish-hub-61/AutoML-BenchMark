/* pages/new-benchmark.js — configuration form, cost estimate, experiment start. */

const { apiGet, apiPost, qp } = ACBS;
const { esc, badge, fmtNum, fmtDur } = window.ui;

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

let datasets = [];
let frameworks = [];

async function loadDatasets() {
  datasets = await apiGet("/datasets");
  const sel = document.getElementById("dataset");
  if (!datasets.length) {
    sel.innerHTML = "<option value=''>No datasets</option>";
    sel.disabled = true;
    document.getElementById("dataset-hint").innerHTML =
      'No datasets yet — <a href="datasets.html">add one first</a>.';
    return;
  }
  sel.innerHTML = datasets
    .map(
      (d) =>
        `<option value="${d.id}" data-name="${esc(d.name)}">${esc(d.name)} · ${d.n_rows}×${d.n_features}</option>`
    )
    .join("");
  const preName = qp("dataset");
  const preId = qp("dataset_id");
  if (preId && datasets.some((d) => String(d.id) === preId)) sel.value = preId;
  else if (preName) {
    const hit = datasets.find((d) => d.name === preName);
    if (hit) sel.value = String(hit.id);
  }
  document.getElementById("dataset-hint").textContent =
    "The target column is detected automatically unless specified.";
}

async function loadFrameworks() {
  frameworks = await apiGet("/frameworks").catch(() => []);
  const box = document.getElementById("framework-list");
  if (!frameworks.length) {
    box.innerHTML = '<p class="small faint">No frameworks registered.</p>';
    return;
  }
  box.innerHTML = frameworks
    .map((f) => {
      const ok = f.installed && f.os_supported;
      const state = ok
        ? badge("installed")
        : `<span class="badge muted">${esc(f.reason || (f.os_supported ? "not installed" : "unsupported OS"))}</span>`;
      return `
        <label class="check" style="margin: 0">
          <input type="checkbox" name="fw" value="${esc(f.id)}" ${ok ? "checked" : ""} ${ok ? "" : "disabled"} />
          <span>
            <span class="t">${esc(f.name)}</span> ${state}<br />
            <span class="small faint">${esc(f.strategy)}</span>
          </span>
        </label>`;
    })
    .join("");
  box.querySelectorAll('input[name="fw"]').forEach((cb) =>
    cb.addEventListener("change", refreshEstimate)
  );
}

function renderWeights() {
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
      refreshEstimate();
    })
  );
}

function readPreferences() {
  const prefs = {};
  document.querySelectorAll("#weights input[type=range]").forEach((r) => {
    prefs[r.dataset.w] = +r.value;
  });
  prefs.method = document.getElementById("method").value;
  return prefs;
}

function readAutoml() {
  const mode = document.querySelector('input[name="fw-mode"]:checked').value;
  const budget_s = Math.max(5, +document.getElementById("budget").value || 60);
  if (mode === "auto") return { auto: true, budget_s };
  const picked = Array.from(document.querySelectorAll('input[name="fw"]:checked')).map(
    (cb) => cb.value
  );
  return { frameworks: picked, budget_s };
}

function readConfig() {
  return {
    models: document.getElementById("classic").checked ? ["all"] : [],
    cv_folds: Math.min(10, Math.max(1, +document.getElementById("folds").value || 3)),
    seed: +document.getElementById("seed").value || 42,
    metrics_primary: document.getElementById("metric").value,
    tuning: "none",
  };
}

async function refreshEstimate() {
  try {
    const manual = document.querySelector('input[name="fw-mode"]:checked').value === "manual";
    const picked = manual
      ? Array.from(document.querySelectorAll('input[name="fw"]:checked')).map((cb) => cb.value)
      : frameworks.filter((f) => f.installed && f.os_supported).map((f) => f.id);
    const est = await apiPost("/frameworks/estimate", {
      automl_frameworks: picked,
      automl_budget_s: Math.max(5, +document.getElementById("budget").value || 60),
      models: readConfig().models,
      n_datasets: 1,
    });
    document.getElementById("estimate").textContent = fmtDur(est.total_wall_seconds);
  } catch (e) {
    document.getElementById("estimate").textContent = "—";
  }
}

function wireForm() {
  document.querySelectorAll('input[name="fw-mode"]').forEach((r) =>
    r.addEventListener("change", () => {
      document.getElementById("framework-list").style.display =
        r.value === "manual" && r.checked ? "flex" : "none";
      refreshEstimate();
    })
  );
  document.querySelectorAll("#budget-presets button").forEach((b) =>
    b.addEventListener("click", () => {
      document.getElementById("budget").value = b.dataset.s;
      refreshEstimate();
    })
  );
  ["folds", "seed", "metric", "classic", "budget"].forEach((id) =>
    document.getElementById(id).addEventListener("change", refreshEstimate)
  );
  document.getElementById("method").addEventListener("change", refreshEstimate);

  document.getElementById("bench-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = document.getElementById("start-btn");
    const datasetId = +document.getElementById("dataset").value;
    if (!datasetId) {
      toast("Choose a dataset first", "error");
      return;
    }
    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      const res = await apiPost("/experiments", {
        dataset_id: datasetId,
        config: readConfig(),
        preferences: readPreferences(),
        automl: readAutoml(),
      });
      const picked = res.automl_auto
        ? (res.automl_selection && res.automl_selection.frameworks.join(", ")) || "auto"
        : (res.automl_frameworks || []).join(", ");
      toast(`Experiment #${res.id} queued: ${picked || "no frameworks"}`, "ok");
      window.location.href = `run-monitor.html?id=${res.id}`;
    } catch (e) {
      toast(e.message, "error");
      btn.disabled = false;
      btn.textContent = "Start benchmark";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderWeights();
  wireForm();
  Promise.all([loadDatasets(), loadFrameworks()])
    .then(refreshEstimate)
    .catch((e) => toast(e.message, "error"));
});
