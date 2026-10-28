/* pages/history.js — experiment list, open/cancel/delete. */

const { apiGet, apiDelete } = ACBS;
const { esc, badge, tableHtml, emptyState, fmtTime, relativeTime } = window.ui;

function openHref(r) {
  if (r.status === "completed") return `results.html?id=${r.id}`;
  return `run-monitor.html?id=${r.id}`;
}

async function load() {
  const rows = (await apiGet("/experiments").catch(() => [])).slice().reverse();
  const box = document.getElementById("table");
  if (!rows.length) {
    box.innerHTML = emptyState(
      "No experiments yet",
      "Start your first benchmark — it takes about a minute.",
      '<a class="btn primary mt-4" href="new-benchmark.html">New benchmark</a>'
    );
    return;
  }
  const cols = [
    {
      key: "id",
      label: "#",
      fmt: (r) => `<a href="${openHref(r)}"><strong>#${r.id}</strong></a>`,
    },
    { key: "status", label: "Status", fmt: (r) => badge(r.status) },
    {
      key: "dataset_id",
      label: "Dataset",
      fmt: (r) => `<span class="faint">dataset #${r.dataset_id}</span>`,
    },
    {
      key: "started_at",
      label: "Started",
      fmt: (r) => `${esc(fmtTime(r.started_at))} <span class="faint">(${esc(relativeTime(r.started_at))})</span>`,
    },
    {
      key: "id",
      label: "",
      cls: "num",
      fmt: (r) => `<button class="btn sm ghost" data-open="${r.id}">Open</button>
                   <button class="btn sm danger" data-del="${r.id}">Delete</button>`,
    },
  ];
  box.innerHTML = tableHtml(cols, rows);

  box.querySelectorAll("[data-open]").forEach((b) =>
    b.addEventListener("click", () => {
      const row = rows.find((r) => String(r.id) === b.dataset.open);
      window.location.href = openHref(row);
    })
  );
  box.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      const rid = b.dataset.del;
      b.disabled = true;
      try {
        const res = await apiDelete(`/experiments/${rid}`);
        if (res.action === "cancelling") {
          toast(`#${rid}: cancellation requested`, "ok");
        } else {
          toast(`#${rid} deleted`, "ok");
          load();
        }
      } catch (e) {
        toast(e.message, "error");
        b.disabled = false;
      }
    })
  );
}

document.addEventListener("DOMContentLoaded", () =>
  load().catch((e) => toast(e.message, "error"))
);
