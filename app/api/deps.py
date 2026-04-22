"""Auth + rate limiting dependencies."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.config import get_settings
from app.core.security import decode_token


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
    except Exception as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from e
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing sub")
    return CurrentUser(id=str(sub), email=payload.get("email"))


# In-memory token bucket per user. Replace with Redis-backed limiter for HA.
_buckets: dict[str, list[float]] = defaultdict(list)


def rate_limit(max_per_min: int | None = None):
    cap = max_per_min or get_settings().rate_limit_per_min

    def _dep(user: CurrentUser = Depends(current_user)) -> CurrentUser:
        now = time.time()
        window_start = now - 60.0
        history = _buckets[user.id]
        # drop entries older than the window
        history[:] = [t for t in history if t >= window_start]
        if len(history) >= cap:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Rate limit exceeded ({cap}/min)",
            )
        history.append(now)
        return user

    return _dep
