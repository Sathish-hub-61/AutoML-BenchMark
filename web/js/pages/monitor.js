/* pages/monitor.js — poll experiment status and render the framework grid. */

const { apiGet, apiDelete, qp } = ACBS;
const { esc, badge, fmtSec, fmtDur, emptyState } = window.ui;

const experimentId = qp("id");
let pollTimer = null;

async function poll() {
  if (!experimentId) return;
  let st;
  try {
    st = await apiGet(`/experiments/${experimentId}/status`);
  } catch (e) {
    document.getElementById("status-line").textContent = `Error: ${e.message}`;
    clearInterval(pollTimer);
    return;
  }

  document.title = `Run #${st.id} (${st.status}) — ACBS`;
  document.getElementById("monitor-sub").textContent = `Experiment #${st.id}`;
  document.getElementById("status-line").innerHTML = `${badge(st.status)} <span class="dim">experiment #${st.id}</span>`;
  document.getElementById("message-line").textContent = st.progress?.message || "";
  document.getElementById("elapsed-line").textContent = st.elapsed_time_s != null ? `elapsed ${fmtDur(st.elapsed_time_s)}` : "";
  const pct = st.progress?.pct ?? 0;
  document.getElementById("progress-bar").style.width = `${pct}%`;

  const cancelBtn = document.getElementById("cancel-btn");
  const canCancel = ["queued", "running"].includes(st.status);
  cancelBtn.style.display = canCancel ? "" : "none";

  const viewBtn = document.getElementById("view-results");
  viewBtn.style.display = st.status === "completed" ? "" : "none";
  if (st.status === "completed") viewBtn.href = `results.html?id=${st.id}`;

  const errBox = document.getElementById("error-box");
  errBox.innerHTML = st.error
    ? `<div class="notice danger"><strong>Reported error:</strong> ${esc(st.error)}</div>`
    : "";

  const fw = st.per_framework_status || {};
  const grid = document.getElementById("monitor-grid");
  if (!Object.keys(fw).length) {
    grid.innerHTML = emptyState(
      "Waiting for the first framework",
      "Folds are prepared first; framework cells appear as they start."
    );
  } else {
    grid.innerHTML = Object.entries(fw)
      .map(([name, s]) => {
        const elapsed = s.elapsed_s != null ? `${fmtSec(s.elapsed_s)}` : "—";
        const budget = s.budget_s ? ` / ${fmtSec(s.budget_s)} budget` : "";
        return `
          <div class="monitor-cell">
            <div class="fw">${esc(name)}</div>
            <div>${badge(s.status || "queued")}</div>
            <div class="meta">${elapsed}${budget}</div>
          </div>`;
      })
      .join("");
  }

  if (["completed", "failed", "cancelled"].includes(st.status)) {
    clearInterval(pollTimer);
    if (st.status === "completed") toast(`Experiment #${st.id} completed`, "ok");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!experimentId) {
    document.getElementById("status-line").textContent =
      "No experiment selected — open one from History.";
    return;
  }
  document.getElementById("cancel-btn").addEventListener("click", async () => {
    try {
      const res = await apiDelete(`/experiments/${experimentId}`);
      toast(res.action === "cancelling" ? "Cancellation requested" : "Removed", "ok");
    } catch (e) {
      toast(e.message, "error");
    }
  });
  poll();
  pollTimer = setInterval(poll, 2000);
});
