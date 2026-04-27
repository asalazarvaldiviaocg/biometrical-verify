import { useEffect, useState } from "react";
import BiometricCapture from "./components/BiometricCapture.jsx";
import { setToken, getToken } from "./lib/api.js";

export default function App() {
  const [contractId, setContractId] = useState("contract-demo-001");
  const [token, setTok] = useState(getToken());
  const [result, setResult] = useState(null);
  const [showToken, setShowToken] = useState(false);

  // Accept ?token=... from URL: caller can hand a link with the dev JWT
  // baked in, the field gets populated automatically, and the param is
  // stripped from the URL bar so it doesn't leak into history.
  useEffect(() => {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get("token");
    if (fromUrl) {
      const trimmed = fromUrl.trim();
      setToken(trimmed);
      setTok(trimmed);
      url.searchParams.delete("token");
      window.history.replaceState({}, "", url.toString());
    }
  }, []);

  const saveToken = (e) => {
    const v = e.target.value.trim();
    setToken(v);
    setTok(v);
  };

  const clearToken = () => {
    setToken("");
    setTok("");
    setResult(null);
  };

  return (
    <div className="shell">
      <h1>Biometrical Verify</h1>
      <p className="subtle">
        Verificación biométrica de identidad con liveness y receipt firmado.
      </p>

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <label>JWT (dev)</label>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={() => setShowToken((s) => !s)}
              style={{ fontSize: 12, padding: "2px 8px" }}
            >
              {showToken ? "Ocultar" : "Mostrar"}
            </button>
            <button
              type="button"
              onClick={clearToken}
              style={{ fontSize: 12, padding: "2px 8px" }}
              disabled={!token}
            >
              Limpiar
            </button>
          </div>
        </div>
        <input
          type={showToken ? "text" : "password"}
          value={token}
          onChange={saveToken}
          placeholder="Pega un JWT emitido por tu API"
          autoComplete="off"
          spellCheck="false"
          style={{ width: "100%", padding: "0.5rem", marginTop: 4, fontFamily: "monospace", fontSize: 12 }}
        />
        <p className="subtle" style={{ marginTop: 4, fontSize: 11 }}>
          {token ? `Token presente (${token.length} caracteres)` : "Sin token"}
        </p>

        <label style={{ display: "block", marginTop: "0.75rem" }}>Contract ID</label>
        <input
          value={contractId}
          onChange={(e) => setContractId(e.target.value)}
          style={{ width: "100%", padding: "0.5rem", marginTop: 4 }}
        />
      </div>

      <div className="card">
        {token ? (
          <BiometricCapture contractId={contractId} onResult={setResult} />
        ) : (
          <p className="subtle">Configura un JWT arriba para iniciar.</p>
        )}
      </div>

      {result && (
        <div className="card">
          <h2 style={{ marginTop: 0 }}>
            Resultado{" "}
            {result.ok && result.decision && (
              <span className={`badge ${result.decision}`}>{result.decision}</span>
            )}
          </h2>
          <pre className="result">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
