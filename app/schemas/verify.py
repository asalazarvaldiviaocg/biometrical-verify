from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["APPROVE", "REVIEW", "REJECT"]
JobStatus = Literal["queued", "running", "done", "error"]
ChallengeName = Literal["blink_twice", "blink_once", "turn_head", "none"]


class ChallengeResponse(BaseModel):
    challenge: ChallengeName
    instruction: str
    nonce: str
    expires_at: str


class VerifyAccepted(BaseModel):
    job_id: str
    status: JobStatus = "queued"
    sha256_id: str
    sha256_video: str


class VerifyResult(BaseModel):
    job_id: str
    status: JobStatus
    decision: Decision | None = None
    similarity: float | None = None
    distance: float | None = None
    threshold: float | None = None
    liveness_score: float | None = None
    challenge_passed: bool | None = None
    deepfake_suspicious: bool | None = None
    model: str | None = None
    receipt_hash: str | None = None
    receipt_signature: str | None = None
    reason: str | None = None
    created_at: str | None = None
    finished_at: str | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
