from __future__ import annotations

import base64
import re
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known-bad placeholder values that ship in the example file. Boot must reject
# these in production rather than silently accept them.
_BAD_DEFAULTS_JWT = {"", "dev-jwt-secret-change-me", "change-me", "secret"}
_BAD_DEFAULTS_MASTER_KEY_DECODED = b"dev-only-master-key-change-me-32-bytes"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Service
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    # DB / cache
    database_url: str = "postgresql+psycopg://bio:bio@db:5432/bio"
    redis_url: str = "redis://redis:6379/0"

    # Object store
    s3_endpoint: str = "http://minio:9000"
    s3_region: str = "us-east-1"
    s3_bucket: str = "bio-blobs"
    s3_access_key: str = "biominio"
    s3_secret_key: str = "biominio-secret"
    s3_use_ssl: bool = False

    # Crypto
    kms_backend: Literal["local", "aws"] = "local"
    master_key: str = ""
    aws_kms_key_id: str = ""
    receipt_signing_key: str = ""

    # Auth
    jwt_secret: str = ""
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 60
    jwt_issuer: str = "biometrical"
    jwt_audience: str = "biometrical-verify"

    # Verification policy
    match_threshold: float = 0.68
    liveness_min_score: float = 0.85
    approve_similarity_min: float = 0.55      # was 0.40 — aligned with ArcFace LFW threshold
    # Was 0.40. Narrows the gray zone: 0.45-0.55 → REVIEW; <0.45 → REJECT.
    review_similarity_min: float = 0.45
    blob_retention_days: int = 30
    rate_limit_per_min: int = 5
    max_id_bytes: int = 8 * 1024 * 1024
    max_video_bytes: int = 25 * 1024 * 1024

    # Models
    anti_spoof_model_path: str = "models/anti_spoof_mn3.onnx"
    anti_spoof_model_sha256: str = ""

    face_model_name: str = "ArcFace"
    face_detector: str = "retinaface"

    @field_validator("jwt_alg")
    @classmethod
    def _reject_none_alg(cls, v: str) -> str:
        # PyJWT 2.x already refuses unsigned tokens by default, but if a
        # caller passes algorithms=[s.jwt_alg] with jwt_alg="none" the check
        # is bypassed. Block the misconfiguration at the source.
        if v.strip().lower() in {"none", ""}:
            raise ValueError(
                "JWT_ALG must be a real signature algorithm (HS256/RS256/...), not 'none' or empty"
            )
        return v

    @field_validator("master_key")
    @classmethod
    def _validate_master_key(cls, v: str) -> str:
        if not v:
            return v
        try:
            raw = base64.b64decode(v)
        except Exception as e:
            raise ValueError("MASTER_KEY must be base64") from e
        if len(raw) != 32:
            raise ValueError("MASTER_KEY must decode to 32 bytes (AES-256)")
        if raw == _BAD_DEFAULTS_MASTER_KEY_DECODED:
            raise ValueError(
                "MASTER_KEY is the documented placeholder — generate a fresh one with `make seed`"
            )
        return v

    @field_validator("anti_spoof_model_sha256")
    @classmethod
    def _validate_model_hash(cls, v: str) -> str:
        if not v:
            return v
        if not re.fullmatch(r"[0-9a-fA-F]{64}", v):
            raise ValueError("ANTI_SPOOF_MODEL_SHA256 must be a 64-char hex string")
        return v.lower()

    @model_validator(mode="after")
    def _enforce_prod_invariants(self) -> Settings:
        if self.app_env != "production":
            return self
        problems: list[str] = []
        if self.jwt_secret in _BAD_DEFAULTS_JWT or len(self.jwt_secret) < 32:
            problems.append("JWT_SECRET must be set to ≥32 chars of entropy in production")
        if not self.master_key and self.kms_backend == "local":
            problems.append("MASTER_KEY must be set when KMS_BACKEND=local")
        if self.kms_backend == "local":
            problems.append(
                "KMS_BACKEND=local is dev-only; switch to KMS_BACKEND=aws + AWS_KMS_KEY_ID for prod"
            )
        if self.kms_backend == "aws" and not self.aws_kms_key_id:
            problems.append("AWS_KMS_KEY_ID must be set when KMS_BACKEND=aws")
        if not self.receipt_signing_key:
            problems.append(
                "RECEIPT_SIGNING_KEY must be set in production — autogeneration would invalidate "
                "every previously-issued receipt across restarts"
            )
        if not self.anti_spoof_model_sha256:
            problems.append(
                "ANTI_SPOOF_MODEL_SHA256 must be set in production "
                "so the loaded model is verifiable"
            )
        if "*" in self.cors_origin_list:
            problems.append("CORS_ORIGINS=* is incompatible with allow_credentials=True")
        if problems:
            joined = "\n  - ".join(problems)
            raise ValueError(
                f"Refusing to boot in production with insecure configuration:\n  - {joined}"
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
