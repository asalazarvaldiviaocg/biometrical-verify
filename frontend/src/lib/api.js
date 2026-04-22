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

export async function getChallenge() {
  const r = await fetch(`${BASE}/verify/challenge`, { headers: authHeaders() });
  if (!r.ok) throw new Error(`challenge failed: ${r.status}`);
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
  if (!r.ok) throw new Error(`submit failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function pollResult(jobId, { interval = 1000, timeoutMs = 60_000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const r = await fetch(`${BASE}/verify/${jobId}`, { headers: authHeaders() });
    if (r.ok) {
      const body = await r.json();
      if (body.status === "done" || body.status === "error") return body;
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
