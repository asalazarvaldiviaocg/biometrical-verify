# Compliance Notes

> Not legal advice. Consult counsel for your jurisdiction.

## GDPR (EU)

- **Article 9** — Biometric data for unique identification is "special
  category" personal data. Lawful basis is required *in addition* to
  Article 6 — usually **explicit consent** (Art. 9(2)(a)).
- **Article 35** — A Data Protection Impact Assessment is mandatory for
  systematic biometric processing. Document risk and mitigation.
- **Article 32** — Encryption + pseudonymisation of personal data at rest
  and in transit. AES-256-GCM with KMS-managed keys satisfies "state of
  the art".
- **Article 17** — Right to erasure: implemented via crypto-shred.
- **Article 20** — Data portability: `GET /api/v1/verify/{job_id}` returns
  the user's verification record + signed receipt.

## LFPDPPP / LGPDPPSO (México)

- Datos biométricos = **datos personales sensibles** (Art. 3 LFPDPPP /
  Art. 3 LGPDPPSO). Consentimiento expreso por escrito requerido.
- Aviso de privacidad integral debe declarar finalidad biométrica,
  responsable, transferencias y derechos ARCO.
- Medidas de seguridad físicas, técnicas y administrativas obligatorias
  (Art. 19). El esquema de cifrado y minimización aquí descrito ayuda a
  cumplir.

## Consent UX (suggested copy)

> Para firmar este contrato verificamos tu identidad mediante una foto de
> tu identificación oficial y un breve video selfie. Procesamos los datos
> únicamente con esta finalidad, los conservamos cifrados durante
> {{retention_days}} días y luego los eliminamos. Puedes ejercer tus
> derechos ARCO en {{contact_email}}.
>
> [ ] Autorizo el tratamiento de mis datos biométricos para verificación
> de identidad en este contrato.

## Retention

`BLOB_RETENTION_DAYS` (default 30) governs raw blob purge. Decision
records and signed receipts are retained per your contractual / legal
basis (typically the lifetime of the contract + statute of limitations).

## Data subject rights

| Right | Implementation |
|-------|----------------|
| Access | `GET /api/v1/verify/{job_id}` |
| Rectification | Re-run verification; old record retained for audit |
| Erasure | Crypto-shred KEK; mark records erased |
| Portability | Return record JSON + signed receipt |
| Objection | Block further submissions for the user |

## Audit log

Every pipeline stage emits an `AuditEvent` row. Operators have read-only
access to events but **never** to raw biometric blobs.
