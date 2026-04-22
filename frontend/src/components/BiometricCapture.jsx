import { useEffect, useRef, useState } from "react";
import { getChallenge, sha256, submitVerification, pollResult } from "../lib/api.js";

const VIDEO_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: "user",
    width: { ideal: 720 },
    height: { ideal: 720 },
    frameRate: { ideal: 24, max: 30 },
  },
};

const MIME = (() => {
  if (typeof MediaRecorder === "undefined") return "video/webm";
  if (MediaRecorder.isTypeSupported("video/webm;codecs=vp9")) return "video/webm;codecs=vp9";
  if (MediaRecorder.isTypeSupported("video/webm;codecs=vp8")) return "video/webm;codecs=vp8";
  return "video/webm";
})();

const RECORD_MS = 4500;

export default function BiometricCapture({ contractId, onResult }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [phase, setPhase] = useState("init");
  const [idFile, setIdFile] = useState(null);
  const [challenge, setChallenge] = useState(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        const ch = await getChallenge();
        setChallenge(ch);
        setPhase("ready");
      } catch (e) {
        console.error(e);
        setPhase("error");
        onResult?.({ ok: false, error: e.message });
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const start = () => {
    if (!idFile) return;
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current, {
      mimeType: MIME,
      videoBitsPerSecond: 1_500_000,
    });
    rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
    rec.onstop = upload;
    recorderRef.current = rec;
    rec.start(250);
    setPhase("recording");

    const t0 = Date.now();
    const tick = setInterval(() => {
      const p = Math.min(1, (Date.now() - t0) / RECORD_MS);
      setProgress(p);
      if (p >= 1) clearInterval(tick);
    }, 100);

    setTimeout(() => rec.state === "recording" && rec.stop(), RECORD_MS);
  };

  const upload = async () => {
    setPhase("uploading");
    try {
      const blob = new Blob(chunksRef.current, { type: MIME });
      const hash = await sha256(blob);
      const { job_id } = await submitVerification({
        contractId,
        challenge: challenge.challenge,
        nonce: challenge.nonce,
        idFile,
        videoBlob: blob,
        videoSha256: hash,
      });
      setPhase("processing");
      const result = await pollResult(job_id);
      setPhase("done");
      onResult?.({ ok: true, ...result });
    } catch (e) {
      console.error(e);
      setPhase("error");
      onResult?.({ ok: false, error: e.message });
    }
  };

  return (
    <div>
      <label className="dropzone">
        {idFile ? `ID seleccionada: ${idFile.name}` : "Selecciona la foto de tu identificación"}
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => setIdFile(e.target.files?.[0] ?? null)}
          disabled={phase !== "ready"}
        />
      </label>

      <video ref={videoRef} muted playsInline className="preview" />

      {challenge && phase !== "done" && (
        <div className="challenge">{challenge.instruction}</div>
      )}

      <div style={{ textAlign: "center" }}>
        <button onClick={start} disabled={phase !== "ready" || !idFile}>
          {phase === "recording" ? `Grabando ${(progress * 100) | 0}%`
            : phase === "uploading" ? "Enviando…"
            : phase === "processing" ? "Verificando…"
            : phase === "done" ? "Listo"
            : "Iniciar verificación"}
        </button>
      </div>

      <Status phase={phase} />
    </div>
  );
}

function Status({ phase }) {
  const map = {
    init: "Inicializando cámara…",
    ready: "Cámara lista.",
    recording: "Grabando — sigue la indicación.",
    uploading: "Subiendo video cifrado…",
    processing: "El servidor está verificando…",
    done: "Verificación completa.",
    error: "Ocurrió un error. Revisa la consola.",
  };
  return <p className="status">{map[phase]}</p>;
}
