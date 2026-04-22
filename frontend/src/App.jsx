import { useState } from "react";
import BiometricCapture from "./components/BiometricCapture.jsx";
import { setToken, getToken } from "./lib/api.js";

export default function App() {
  const [contractId, setContractId] = useState("contract-demo-001");
  const [token, setTok] = useState(getToken());
  const [result, setResult] = useState(null);

  const saveToken = (e) => {
    setToken(e.target.value);
    setTok(e.target.value);
  };

  return (
    <div className="shell">
      <h1>Biometrical Verify</h1>
      <p className="subtle">
        Verificación biométrica de identidad con liveness y receipt firmado.
      </p>

      <div className="card">
        <label>JWT (dev)</label>
        <input
          type="password"
          value={token}
          onChange={saveToken}
          placeholder="Pega un JWT emitido por tu API"
          style={{ width: "100%", padding: "0.5rem", marginTop: 4 }}
        />

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
