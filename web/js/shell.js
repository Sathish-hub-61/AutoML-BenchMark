/* shell.js — shared navigation shell, health dot, toasts, tabs. */

const NAV = [
  ["home", "Home", "index.html"],
  ["datasets", "Datasets", "datasets.html"],
  ["new", "New Benchmark", "new-benchmark.html"],
  ["monitor", "Run Monitor", "run-monitor.html"],
  ["results", "Results", "results.html"],
  ["history", "History", "history.html"],
  ["frameworks", "Frameworks", "frameworks.html"],
  ["methodology", "Methodology", "methodology.html"],
];

function mountShell() {
  const page = document.body.dataset.page || "";
  const header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML = `
    <div class="inner">
      <a class="brand" href="index.html">
        <span class="mark"></span>ACBS <span class="tag">v3</span>
      </a>
      <nav class="site-nav">
        ${NAV.map(
          ([id, label, href]) =>
            `<a href="${href}" class="${id === page ? "active" : ""}">${label}</a>`
        ).join("")}
      </nav>
      <span class="spacer"></span>
      <span class="api-state" id="api-state" title="API health"><span class="dot"></span><span id="api-state-text">…</span></span>
    </div>`;
  document.body.prepend(header);

  const footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML = `
    <div class="inner">
      <span>Auto Classifier Benchmarking System — AMLB-style protocol, identical folds, central scoring.</span>
      <span class="mono" id="foot-version"></span>
    </div>`;
  document.body.append(footer);

  const root = document.createElement("div");
  root.id = "toast-root";
  document.body.append(root);

  checkHealth();
}

async function checkHealth() {
  const el = document.getElementById("api-state");
  const txt = document.getElementById("api-state-text");
  try {
    const h = await ACBS.apiGet("/health");
    el.classList.add("up");
    txt.textContent = "API up";
    const v = (h && (h.version || h.lib_version)) || "";
    const foot = document.getElementById("foot-version");
    if (foot && v) foot.textContent = `api ${v}`;
  } catch {
    el.classList.remove("up");
    txt.textContent = "API down";
  }
}

/* ---------- toasts ---------- */

function toast(message, kind = "info", ms = 4200) {
  let root = document.getElementById("toast-root");
  if (!root) {
    root = document.createElement("div");
    root.id = "toast-root";
    document.body.append(root);
  }
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = message;
  root.append(t);
  setTimeout(() => t.remove(), ms);
}

/* ---------- tabs ---------- */

function wireTabs(scope) {
  (scope || document).querySelectorAll(".tabs").forEach((tabs) => {
    const buttons = tabs.querySelectorAll("button[data-tab]");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        buttons.forEach((b) => b.classList.toggle("active", b === btn));
        const wrap = tabs.parentElement;
        wrap.querySelectorAll(":scope > .tab-panel").forEach((p) => {
          p.classList.toggle("active", p.id === btn.dataset.tab);
        });
      });
    });
  });
}

window.shell = { toast, wireTabs, checkHealth, mountShell };
document.addEventListener("DOMContentLoaded", mountShell);
