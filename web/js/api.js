/* api.js — tiny fetch wrapper for the ACBS API (mounted at the root). */

(function () {
  const API_BASE = "";

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function _handle(res) {
  if (res.status === 204) return null;
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const detail =
      (data && (data.detail || data.message)) || (typeof data === "string" ? data : "") || res.statusText;
    throw new ApiError(detail || `HTTP ${res.status}`, res.status, data);
  }
  return data;
}

async function apiGet(path, params) {
  const url = new URL(API_BASE + path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
  }
  const res = await fetch(url.toString(), { headers: { Accept: "application/json" } });
  return _handle(res);
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return _handle(res);
}

async function apiDelete(path) {
  const res = await fetch(API_BASE + path, { method: "DELETE" });
  return _handle(res);
}

  async function apiUpload(path, formData) {
    const res = await fetch(API_BASE + path, { method: "POST", body: formData });
    return _handle(res);
  }

  /** Read ?id= and friends from the current URL. */
  function qp(name, fallback = null) {
    const v = new URLSearchParams(window.location.search).get(name);
    return v === null ? fallback : v;
  }

  window.ACBS = { apiGet, apiPost, apiDelete, apiUpload, qp, ApiError };
})();
