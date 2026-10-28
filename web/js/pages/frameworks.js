/* pages/frameworks.js — framework registry with install state and badges. */

const { apiGet } = ACBS;
const { esc, badge, tableHtml, emptyState } = window.ui;

async function load() {
  const rows = await apiGet("/frameworks");
  const box = document.getElementById("table");
  if (!rows || !rows.length) {
    box.innerHTML = emptyState("No frameworks registered", "Check configs/frameworks_info.yaml.");
    return;
  }
  const cols = [
    {
      key: "name",
      label: "Framework",
      fmt: (r) => `<strong>${esc(r.name)}</strong> <span class="faint mono small">${esc(r.id)}</span>`,
    },
    {
      key: "strategy",
      label: "Strategy",
      fmt: (r) => `<span class="small">${esc(r.strategy)}</span>`,
    },
    {
      key: "installed",
      label: "Environment",
      fmt: (r) =>
        r.installed
          ? '<span class="badge installed">installed</span>'
          : '<span class="badge muted">not installed</span>',
    },
    {
      key: "version",
      label: "Version",
      cls: "num",
      fmt: (r) => (r.version ? `<span class="mono">${esc(r.version)}</span>` : '<span class="faint">—</span>'),
    },
    {
      key: "os_supported",
      label: "OS",
      fmt: (r) =>
        r.os_supported
          ? '<span class="badge ok">supported</span>'
          : `<span class="badge failed">${esc(r.reason || "unsupported")}</span>`,
    },
    {
      key: "default_budget_s",
      label: "Default budget",
      cls: "num",
      fmt: (r) => `${r.default_budget_s}s`,
    },
    {
      key: "id",
      label: "",
      cls: "num",
      fmt: (r) =>
        r.installed && r.os_supported
          ? '<a class="btn sm" href="new-benchmark.html">Benchmark</a>'
          : `<code class="mono faint small">setup_envs.py --framework ${esc(r.id)}</code>`,
    },
  ];
  box.innerHTML = tableHtml(cols, rows);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refresh").addEventListener("click", () =>
    load().catch((e) => toast(e.message, "error"))
  );
  load().catch((e) => toast(e.message, "error"));
});
