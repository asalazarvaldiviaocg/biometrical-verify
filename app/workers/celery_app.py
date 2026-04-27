from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

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

# Periodic background tasks. Run a Celery beat process alongside the worker
# (`celery -A app.workers.celery_app.celery beat`) — the docker-compose stack
# already includes a `beat` service for this.
celery.conf.beat_schedule = {
    "requeue-stuck-verifications": {
        "task":     "verify.requeue_stuck",
        "schedule": 60.0,            # every 60 s — aligns with the 5-min cutoff
        "options":  {"queue": "verify"},
    },
    "purge-expired-nonces": {
        "task":     "verify.purge_expired_nonces",
        "schedule": crontab(minute="*/15"),
        "options":  {"queue": "verify"},
    },
    "purge-revoked-keks": {
        "task":     "verify.purge_revoked_keks",
        # Daily 03:17 UTC — middle-of-night, off main traffic window
        "schedule": crontab(hour=3, minute=17),
        "options":  {"queue": "verify"},
    },
}
