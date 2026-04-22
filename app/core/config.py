from __future__ import annotations

import base64
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    jwt_secret: str = "dev-jwt-secret-change-me"
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 60

    # Verification policy
    match_threshold: float = 0.68
    liveness_min_score: float = 0.85
    approve_similarity_min: float = 0.40
    review_similarity_min: float = 0.32
    blob_retention_days: int = 30
    rate_limit_per_min: int = 5
    max_id_bytes: int = 8 * 1024 * 1024
    max_video_bytes: int = 25 * 1024 * 1024

    # Models
    anti_spoof_model_path: str = "models/anti_spoof_mn3.onnx"
    face_model_name: str = "ArcFace"
    face_detector: str = "retinaface"

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
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_prod(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
