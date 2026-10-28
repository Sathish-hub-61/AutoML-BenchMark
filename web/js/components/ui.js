/* components/ui.js — DOM helpers, formatters, badges, tables. */

(function () {
function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  });
  (Array.isArray(children) ? children : [children]).forEach((c) => {
    if (c === null || c === undefined) return;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  });
  return node;
}

/* ---------- formatters ---------- */

function fmtSec(s) {
  if (s === null || s === undefined || Number.isNaN(+s)) return "—";
  const n = +s;
  if (n < 1) return `${n.toFixed(2)}s`;
  if (n < 90) return `${n.toFixed(1)}s`;
  const m = Math.floor(n / 60);
  const rest = Math.round(n % 60);
  if (m < 60) return `${m}m ${rest}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

function fmtDur(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.max(0, Math.round(+seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(+v)) return "—";
  const n = +v;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(digits);
}

function fmtBytesMb(mb) {
  if (!mb && mb !== 0) return "—";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return String(iso);
  }
}

function relativeTime(iso) {
  if (!iso) return "—";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/* ---------- badges & bits ---------- */

function badge(status) {
  const s = String(status || "unknown").toLowerCase();
  return `<span class="badge ${esc(s)}">${esc(s)}</span>`;
}

function emptyState(title, message, actionHtml = "") {
  return `<div class="empty"><h3>${esc(title)}</h3><p>${esc(message)}</p>${actionHtml}</div>`;
}

function spinner(text = "Loading…") {
  return `<p class="faint small">${esc(text)}</p>`;
}

/** Build an HTML table. columns: [{key,label,cls,fmt}] */
function tableHtml(columns, rows, opts = {}) {
  if (!rows || !rows.length) return opts.empty || emptyState("Nothing here", "No rows to show.");
  const head = columns
    .map((c) => `<th class="${c.cls || ""}">${esc(c.label)}</th>`)
    .join("");
  const body = rows
    .map((r) => {
      const tds = columns
        .map((c) => {
          const raw = c.fmt ? c.fmt(r) : r[c.key];
          return `<td class="${c.cls || ""}">${raw}</td>`;
        })
        .join("");
      return `<tr class="${r._cls || ""}">${tds}</tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

window.ui = {
  esc,
  el,
  badge,
  emptyState,
  spinner,
  tableHtml,
  fmtSec,
  fmtDur,
  fmtNum,
  fmtBytesMb,
  fmtTime,
  relativeTime,
};
})();
