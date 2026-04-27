# syntax=docker/dockerfile:1.7
#
# Two build targets:
#   - `base`        : code + deps, models mounted at runtime via volume (dev / compose).
#   - `production`  : code + deps + pre-baked anti-spoof model, single
#                     immutable image deployable to k8s/ECS/Fly without a
#                     compose volume. Build: `docker build --target production .`
#                     Requires `make seed` to have run first so the model
#                     exists at ./models/anti_spoof_mn3.onnx with the right
#                     hash; otherwise the runtime hash check rejects it.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 bio && chown -R bio:bio /srv
USER bio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ── production: bake the verified anti-spoof model into the image ─────────
# The runtime still re-checks ANTI_SPOOF_MODEL_SHA256 against this file, so
# a swap on the build host is detected at boot. Use this target for any
# deployment without a persistent volume mount (k8s, Fly, Cloud Run, ECS).

FROM base AS production
USER root
COPY --chown=bio:bio models /srv/models
USER bio
