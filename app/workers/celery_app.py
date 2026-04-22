from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

_settings = get_settings()

celery = Celery(
    "biometrical_verify",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["app.workers.tasks"],
)

celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="verify",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_time_limit=120,
    task_soft_time_limit=90,
)
