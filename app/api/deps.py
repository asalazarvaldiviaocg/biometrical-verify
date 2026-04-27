"""Auth + rate limiting dependencies."""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.security import decode_token
from app.services.redis_client import get_redis

# Same allow-list used by storage layer — block traversal early.
_SAFE_SUB = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@dataclass
class CurrentUser:
    id: str
    email: str | None = None


def current_user(authorization: Annotated[str | None, Header()] = None) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except jwt.InvalidAudienceError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong audience") from e
    except jwt.InvalidIssuerError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong issuer") from e
    except jwt.MissingRequiredClaimError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Missing claim: {e.claim}") from e
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired") from e
    except Exception as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from e
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing sub")
    sub = str(sub)
    # Reject sub values that would corrupt S3 keys / cross tenants.
    if not _SAFE_SUB.match(sub):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"Subject must match {_SAFE_SUB.pattern}",
        )
    return CurrentUser(id=sub, email=payload.get("email"))


# ── Rate limit ────────────────────────────────────────────────────────────
#
# Redis-backed sliding window via INCR + EXPIRE on a per-(bucket, user, minute)
# key. With multiple API replicas this gives a single shared counter; the old
# in-memory bucket multiplied the real cap by N replicas.
#
# Falls back to per-process token bucket when Redis is unreachable to avoid
# locking out legitimate users during an infra blip — fail-open is the
# correct trade-off for verification workloads where a missed hit is much
# cheaper than a missed customer.

_fallback: dict[str, list[float]] = defaultdict(list)
_fallback_last_gc: float = 0.0
_fallback_lock = threading.Lock()
# Sweep horizon: a fallback bucket entry whose newest hit is older than this
# many seconds is dropped wholesale. 60 s == the rate-limit window itself, so
# anything beyond that is logically dead and just costs RAM.
_FALLBACK_STALE_S = 60.0
_FALLBACK_GC_EVERY_S = 60.0


def _gc_fallback(now: float) -> None:
    """Periodic sweep so distinct (bucket, user) keys don't accumulate forever
    when Redis is the steady-state path and the fallback only fires during
    blips. Without this the dict grew unboundedly across the process lifetime.
    """
    global _fallback_last_gc
    if now - _fallback_last_gc < _FALLBACK_GC_EVERY_S:
        return
    with _fallback_lock:
        if now - _fallback_last_gc < _FALLBACK_GC_EVERY_S:
            return
        _fallback_last_gc = now
        cutoff = now - _FALLBACK_STALE_S
        dead = [k for k, hist in _fallback.items() if not hist or hist[-1] < cutoff]
        for k in dead:
            _fallback.pop(k, None)


def rate_limit(max_per_min: int | None = None, *, bucket: str = "default"):
    cap = max_per_min or get_settings().rate_limit_per_min

    def _dep(user: CurrentUser = Depends(current_user)) -> CurrentUser:
        if _redis_count_and_check(bucket, user.id, cap):
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Rate limit exceeded ({cap}/min)",
            )
        return user

    return _dep


def _redis_count_and_check(bucket: str, user_id: str, cap: int) -> bool:
    """Returns True iff the request must be rejected (over the cap)."""
    minute_window = int(time.time() // 60)
    key = f"rl:{bucket}:{user_id}:{minute_window}".encode()
    try:
        r = get_redis()
        # Atomic incr; first hit also pins TTL=70s so the key is GCed cleanly.
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 70)
        count, _ = pipe.execute()
        return int(count) > cap
    except RedisError:
        # Fallback: per-process bucket. Better to slightly over-allow during a
        # Redis outage than to wedge every customer.
        now = time.time()
        _gc_fallback(now)
        history = _fallback[f"{bucket}:{user_id}"]
        history[:] = [t for t in history if t >= now - 60.0]
        if len(history) >= cap:
            return True
        history.append(now)
        return False
