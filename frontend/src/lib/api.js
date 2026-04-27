const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

export function setToken(token) {
  localStorage.setItem("bv_token", token);
}

export function getToken() {
  return localStorage.getItem("bv_token") || "";
}

function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}`, ...extra } : { ...extra };
}

// Build a useful error message out of a non-OK response. The API returns
// {"detail": "..."} for validation/auth/policy failures — surface that to
// the user instead of a bare HTTP status code so they can act on it.
async function explainError(prefix, r) {
  let detail = `${r.status}`;
  try {
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await r.json();
      if (j && typeof j.detail === "string") detail = `${r.status} — ${j.detail}`;
      else if (j) detail = `${r.status} — ${JSON.stringify(j)}`;
    } else {
      const text = await r.text();
      if (text) detail = `${r.status} — ${text.slice(0, 200)}`;
    }
  } catch {
    /* response body wasn't parseable; stick with the status code */
  }
  return new Error(`${prefix}: ${detail}`);
}

export async function getChallenge() {
  const r = await fetch(`${BASE}/verify/challenge`, { headers: authHeaders() });
  if (!r.ok) throw await explainError("challenge failed", r);
  return r.json();
}

export async function submitVerification({
  contractId,
  challenge,
  nonce,
  idFile,
  videoBlob,
  videoSha256,
}) {
  const fd = new FormData();
  fd.append("contract_id", contractId);
  fd.append("challenge", challenge);
  fd.append("nonce", nonce);
  fd.append("id_image", idFile);
  fd.append(
    "selfie_video",
    new File([videoBlob], `selfie-${Date.now()}.webm`, { type: videoBlob.type }),
  );

  const r = await fetch(`${BASE}/verify/submit`, {
    method: "POST",
    body: fd,
    headers: authHeaders({ "X-Client-Video-SHA256": videoSha256 }),
  });
  if (!r.ok) throw await explainError("submit failed", r);
  return r.json();
}

export async function pollResult(jobId, { interval = 1000, timeoutMs = 60_000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const r = await fetch(`${BASE}/verify/${jobId}`, { headers: authHeaders() });
    if (r.ok) {
      const body = await r.json();
      if (body.status === "done" || body.status === "error") return body;
    } else if (r.status === 401 || r.status === 403 || r.status === 404) {
      // Permanent failures — no point polling. Surface to caller.
      throw await explainError("status check failed", r);
    }
    await new Promise((res) => setTimeout(res, interval));
  }
  throw new Error("verification timed out");
}

export async function sha256(blob) {
  const buf = await blob.arrayBuffer();
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
