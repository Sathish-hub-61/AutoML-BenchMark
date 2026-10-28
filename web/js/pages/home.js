/* pages/home.js — stats and recent experiments. */

const { apiGet, apiPost } = ACBS;
const { esc, badge, tableHtml, emptyState, fmtSec, relativeTime } = window.ui;

async function loadHealth() {
  try {
    const h = await apiGet("/health");
    document.getElementById("foot-version");
    return h;
  } catch {
    return null;
  }
}

async function loadStats() {
  const [datasets, experiments, frameworks] = await Promise.all([
    apiGet("/datasets").catch(() => []),
    apiGet("/experiments").catch(() => []),
    apiGet("/frameworks").catch(() => []),
  ]);
  const installed = frameworks.filter((f) => f.installed).length;
  const completed = experiments.filter((e) => e.status === "completed").length;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="label">Datasets</div><div class="value">${datasets.length}</div>
      <div class="hint">stored locally</div></div>
    <div class="stat"><div class="label">Experiments</div><div class="value">${experiments.length}</div>
      <div class="hint">${completed} completed</div></div>
    <div class="stat"><div class="label">Frameworks ready</div><div class="value">${installed}/${frameworks.length}</div>
      <div class="hint">isolated environments</div></div>
    <div class="stat"><div class="label">Protocol</div><div class="value">AMLB</div>
      <div class="hint">identical folds · one seed</div></div>`;

  const recent = experiments.slice(-5).reverse();
  const target = document.getElementById("recent");
  if (!recent.length) {
    target.innerHTML = emptyState(
      "No experiments yet",
      "Start your first benchmark to see results here.",
      '<a class="btn primary mt-4" href="new-benchmark.html">New benchmark</a>'
    );
    return;
  }
  target.innerHTML = tableHtml(
    [
      {
        key: "id",
        label: "#",
        fmt: (r) => `<a href="${r.status === "completed" ? `results.html?id=${r.id}` : `run-monitor.html?id=${r.id}`}">#${r.id}</a>`,
      },
      { key: "status", label: "Status", fmt: (r) => badge(r.status) },
      { key: "started_at", label: "When", fmt: (r) => relativeTime(r.started_at) },
    ],
    recent
  );
}

document.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  loadStats().catch((e) => toast(e.message, "error"));
});
