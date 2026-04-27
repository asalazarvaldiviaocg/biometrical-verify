"""Lazy singleton Redis client used by rate limiter, nonce store, and any
other component that needs a shared coordination layer across processes.
"""

from __future__ import annotations

import redis

from app.core.config import get_settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        s = get_settings()
        _client = redis.Redis.from_url(s.redis_url, decode_responses=False)
    return _client
