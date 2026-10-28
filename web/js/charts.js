/* charts.js — Plotly helpers with vendored loader and shared dark template. */

let _plotlyPromise = null;
let _PLOTLY = null;

/** Locate the loaded Plotly: a plain browser attaches it to window, but a page
 *  that happens to expose a CommonJS-like `exports` global sends it there. */
function _findPlotly() {
  if (window.Plotly) return window.Plotly;
  try {
    if (typeof exports === "object" && exports && exports.Plotly) return exports.Plotly;
  } catch {
    /* ignore */
  }
  return null;
}

/** Load web/vendor/plotly.min.js once; falls back to the CDN if vendoring failed. */
function ensurePlotly() {
  const found = _findPlotly();
  if (found) {
    _PLOTLY = found;
    return Promise.resolve(_PLOTLY);
  }
  if (!_plotlyPromise) {
    _plotlyPromise = new Promise((resolve, reject) => {
      const done = () => {
        _PLOTLY = _findPlotly();
        _PLOTLY ? resolve(_PLOTLY) : reject(new Error("Plotly loaded but exported nothing"));
      };
      const s = document.createElement("script");
      s.src = "vendor/plotly.min.js";
      s.onload = done;
      s.onerror = () => {
        const c = document.createElement("script");
        c.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
        c.onload = done;
        c.onerror = () => reject(new Error("Could not load Plotly"));
        document.head.append(c);
      };
      document.head.append(s);
    });
  }
  return _plotlyPromise;
}

const CHART_CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };

function baseLayout(title) {
  return {
    title: title ? { text: title, font: { size: 13, color: "#9aa9bd" } } : undefined,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9aa9bd", size: 12, family: "system-ui, sans-serif" },
    margin: { t: title ? 34 : 16, r: 16, b: 42, l: 52 },
    xaxis: { gridcolor: "#262e3a", zerolinecolor: "#38455a" },
    yaxis: { gridcolor: "#262e3a", zerolinecolor: "#38455a" },
    legend: { orientation: "h", y: -0.25, font: { size: 11 } },
    showlegend: true,
  };
}

async function drawPlot(elId, traces, title) {
  const Plotly = await ensurePlotly();
  const node = typeof elId === "string" ? document.getElementById(elId) : elId;
  if (!node) return;
  const layout = Object.assign(baseLayout(title), { autosize: true });
  await Plotly.newPlot(node, traces, layout, CHART_CONFIG);
}

/* ---------- charts ---------- */

/** Horizontal bars of one metric across models. rows: [{model_name, value, excluded}] */
function chartMetricBars(elId, rows, metricLabel) {
  const sorted = [...rows].sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  const trace = {
    type: "bar",
    orientation: "h",
    y: sorted.map((r) => r.model_name),
    x: sorted.map((r) => r.value),
    marker: {
      color: sorted.map((r) =>
        r.excluded ? "#4b5563" : r.is_rec ? "#3b82f6" : "#60a5fa"
      ),
    },
    text: sorted.map((r) => (r.value == null ? "" : r.value.toFixed(3))),
    textposition: "outside",
    hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
  };
  return drawPlot(elId, [trace], metricLabel).then(() => sorted[0]);
}

/** Pareto scatter: training time vs metric, Pareto front highlighted. */
function chartPareto(elId, rows, metricKey) {
  const pts = rows.filter((r) => r.training_time != null && r[metricKey] != null && !r.excluded);
  const front = paretoFront(pts, "training_time", metricKey);
  const trace = {
    mode: "markers+text",
    type: "scatter",
    x: pts.map((r) => r.training_time),
    y: pts.map((r) => r[metricKey]),
    text: pts.map((r) => r.model_name),
    textposition: "top center",
    textfont: { size: 10 },
    marker: {
      size: 12,
      color: pts.map((r) => (front.includes(r.model_name) ? "#4ade80" : "#60a5fa")),
    },
    hovertemplate: "%{text}<br>train %{x:.1f}s<br>" + metricKey + " %{y:.4f}<extra></extra>",
  };
  const layout = {
    xaxis: { title: "training time (s)", type: "log", gridcolor: "#262e3a" },
    yaxis: { title: metricKey, gridcolor: "#262e3a" },
  };
  return drawPlot(elId, [trace], "Score vs training time (green = Pareto front)").then(async () => {
    const node = document.getElementById(elId);
    const P = await ensurePlotly();
    if (node && P) await P.relayout(node, layout);
  });
}

function paretoFront(rows, xKey, yKey) {
  const front = [];
  [...rows]
    .sort((a, b) => a[xKey] - b[xKey])
    .forEach((r) => {
      if (!front.some((f) => f[yKey] >= r[yKey])) front.push(r);
    });
  return front.map((f) => f.model_name);
}

/** Confusion matrix heatmap from y_true/y_pred and class labels. */
function chartConfusion(elId, yTrue, yPred, classes) {
  const idx = Object.fromEntries(classes.map((c, i) => [c, i]));
  const n = classes.length;
  const m = classes.map(() => new Array(n).fill(0));
  yTrue.forEach((t, i) => {
    const p = yPred[i];
    if (idx[t] === undefined || idx[p] === undefined) return;
    m[idx[t]][idx[p]] += 1;
  });
  const trace = {
    type: "heatmap",
    x: classes,
    y: classes,
    z: m,
    colorscale: [[0, "#151a22"], [1, "#3b82f6"]],
    showscale: false,
    text: m.map((row) => row.map((v) => (v ? String(v) : ""))),
    texttemplate: "%{text}",
    hovertemplate: "true %{y} / pred %{x}: %{z}<extra></extra>",
  };
  return drawPlot(elId, [trace], "Confusion matrix (rows = true, cols = predicted)");
}

/** ROC curves: binary uses column 1; multiclass one-vs-rest per class. */
function chartRoc(elId, yTrue, yProba, classes) {
  if (!yProba || !yProba.length) return Promise.resolve();
  const k = classes.length;
  const traces = [];
  if (k === 2) {
    const pos = classes[1];
    const scores = yProba.map((r) => r[1]);
    const labels = yTrue.map((t) => (t === pos ? 1 : 0));
    const { fpr, tpr, auc } = rocCurve(labels, scores);
    traces.push(rocTrace(fpr, tpr, `pos = ${pos} (AUC ${auc.toFixed(3)})`, "#60a5fa"));
  } else {
    classes.forEach((c, i) => {
      const labels = yTrue.map((t) => (t === c ? 1 : 0));
      const scores = yProba.map((r) => r[i] ?? 0);
      const { fpr, tpr, auc } = rocCurve(labels, scores);
      traces.push(rocTrace(fpr, tpr, `${c} (AUC ${auc.toFixed(3)})`, colorFor(i)));
    });
  }
  traces.push({
    mode: "lines",
    type: "scatter",
    x: [0, 1],
    y: [0, 1],
    line: { dash: "dot", color: "#4b5563", width: 1 },
    showlegend: false,
    hoverinfo: "skip",
  });
  const layout = {
    xaxis: { title: "false positive rate", gridcolor: "#262e3a", range: [0, 1] },
    yaxis: { title: "true positive rate", gridcolor: "#262e3a", range: [0, 1] },
  };
  return drawPlot(elId, traces, "ROC").then(async () => {
    const node = document.getElementById(elId);
    const P = await ensurePlotly();
    if (node && P) await P.relayout(node, layout);
  });
}

function rocTrace(fpr, tpr, name, color) {
  return {
    mode: "lines",
    type: "scatter",
    x: fpr,
    y: tpr,
    name,
    line: { color, width: 2 },
    hovertemplate: "fpr %{x:.3f} / tpr %{y:.3f}<extra></extra>",
  };
}

function rocCurve(labels, scores) {
  const pairs = scores
    .map((s, i) => ({ s, y: labels[i] }))
    .sort((a, b) => b.s - a.s);
  const P = labels.filter((y) => y === 1).length || 1;
  const N = labels.length - P || 1;
  let tp = 0;
  let fp = 0;
  const fpr = [0];
  const tpr = [0];
  let prev = null;
  for (const p of pairs) {
    if (p.y === 1) tp += 1;
    else fp += 1;
    if (prev !== null && p.s !== prev) {
      fpr.push(fp / N);
      tpr.push(tp / P);
    }
    prev = p.s;
  }
  fpr.push(fp / N);
  tpr.push(tp / P);
  // trapezoidal AUC
  let auc = 0;
  for (let i = 1; i < fpr.length; i++) {
    auc += ((fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1])) / 2;
  }
  return { fpr, tpr, auc };
}

/** Calibration curve (reliability diagram) for binary positive class. */
function chartCalibration(elId, yTrue, yProba, classes) {
  if (!yProba || classes.length !== 2) return Promise.resolve();
  const pos = classes[1];
  const bins = Array.from({ length: 10 }, () => ({ n: 0, sum: 0, pos: 0 }));
  yProba.forEach((r, i) => {
    const p = Math.min(0.999, Math.max(0.001, r[1]));
    const b = bins[Math.floor(p * 10)];
    b.n += 1;
    b.sum += p;
    b.pos += yTrue[i] === pos ? 1 : 0;
  });
  const used = bins.filter((b) => b.n > 0);
  const trace = {
    mode: "lines+markers",
    type: "scatter",
    x: used.map((b) => b.sum / b.n),
    y: used.map((b) => b.pos / b.n),
    name: "model",
    line: { color: "#60a5fa", width: 2 },
  };
  const diag = {
    mode: "lines",
    type: "scatter",
    x: [0, 1],
    y: [0, 1],
    name: "perfectly calibrated",
    line: { dash: "dot", color: "#4b5563", width: 1 },
  };
  const layout = {
    xaxis: { title: "mean predicted probability", gridcolor: "#262e3a", range: [0, 1] },
    yaxis: { title: "observed frequency", gridcolor: "#262e3a", range: [0, 1] },
  };
  return drawPlot(elId, [diag, trace], "Calibration").then(async () => {
    const node = document.getElementById(elId);
    const P = await ensurePlotly();
    if (node && P) await P.relayout(node, layout);
  });
}

const PALETTE = ["#60a5fa", "#4ade80", "#fbbf24", "#f87171", "#c084fc", "#22d3ee", "#fb923c", "#a3e635"];
function colorFor(i) {
  return PALETTE[i % PALETTE.length];
}

window.charts = {
  ensurePlotly,
  drawPlot,
  chartMetricBars,
  chartPareto,
  chartConfusion,
  chartRoc,
  chartCalibration,
  colorFor,
};
