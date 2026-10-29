/* pages/datasets.js — stored datasets and CSV upload. */

const { apiGet, apiPost, apiUpload, qp } = ACBS;
const { esc, badge, tableHtml, emptyState } = window.ui;

async function loadDatasets() {
  const rows = await apiGet("/datasets");
  const target = document.getElementById("datasets-table");
  if (!rows.length) {
    target.innerHTML = emptyState("No datasets yet", "Upload a CSV to get started.");
    return;
  }
  target.innerHTML = tableHtml(
    [
      {
        key: "name",
        label: "Name",
        fmt: (r) =>
          `<a href="new-benchmark.html?dataset=${encodeURIComponent(r.name)}" title="Benchmark this dataset">${esc(r.name)}</a>`,
      },
      { key: "source", label: "Source", fmt: (r) => `<span class="faint">${esc(r.source)}</span>` },
      { key: "n_rows", label: "Rows", cls: "num" },
      { key: "n_features", label: "Features", cls: "num" },
      {
        key: "id",
        label: "",
        cls: "num",
        fmt: (r) => `<a class="btn sm" href="new-benchmark.html?dataset_id=${r.id}">Benchmark</a>`,
      },
    ],
    rows
  );
}

function wireUpload() {
  const input = document.getElementById("upload-file");
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    const out = document.getElementById("upload-result");
    out.innerHTML = `<div class="notice">Uploading ${esc(file.name)}…</div>`;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await apiUpload("/datasets", fd);
      const warn = (res.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
      out.innerHTML = `
        <div class="notice warn mt-4">
          <strong>${esc(res.name)}</strong> imported — target <code>${esc(res.target || "?")}</code>
          (${esc(res.target_source || "auto")}).${warn ? `<ul style="margin:6px 0 0 18px">${warn}</ul>` : ""}
        </div>`;
      toast("Dataset uploaded", "ok");
      loadDatasets();
    } catch (e) {
      out.innerHTML = `<div class="notice danger mt-4">${esc(e.message)}</div>`;
    } finally {
      input.value = "";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wireUpload();
  loadDatasets().catch((e) => toast(e.message, "error"));
});
