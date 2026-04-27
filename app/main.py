from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import routes_admin, routes_health, routes_keys, routes_verify
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("startup", env=settings.app_env, version=__version__)
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Biometrical Verify",
        version=__version__,
        description=(
            "Open-source biometric identity verification service. "
            "Face match + liveness + signed receipt for digital contract signing."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Client-Video-SHA256"],
    )
    app.include_router(routes_health.router)
    app.include_router(routes_keys.router)
    app.include_router(routes_verify.router)
    app.include_router(routes_admin.router)
    return app


app = create_app()
