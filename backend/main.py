"""FastAPI application entrypoint.

Run locally with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from backend.api.routes_chat import router as chat_router
from backend.api.routes_health import router as health_router
from backend.api.routes_navigate import router as navigate_router
from backend.api.routes_schemes import router as schemes_router
from backend.api.routes_voice import router as voice_router
from backend.core.config import get_server_settings
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import configure_logging
from backend.core.rate_limit import limiter
from backend.services.scheme_service import get_scheme_service

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Warming up scheme knowledge base and vector index...")
    get_scheme_service()
    logger.info("Startup complete.")
    yield


def create_app() -> FastAPI:
    server_settings = get_server_settings()

    app = FastAPI(
        title="AI Citizen Scheme & Support Navigator",
        description=(
            "Multi-agent API that discovers, verifies, and explains public welfare "
            "entitlements for Indian citizens using open government data."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    if server_settings.backend_api_key:
        @app.middleware("http")
        async def enforce_api_key(request: Request, call_next) -> Response:
            protected = request.url.path.startswith("/api/") and not request.url.path.endswith("/health")
            if protected:
                provided = request.headers.get("X-API-Key", "")
                if provided != server_settings.backend_api_key:
                    return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key header"})
            return await call_next(request)

    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(schemes_router, prefix="/api/v1", tags=["schemes"])
    app.include_router(navigate_router, prefix="/api/v1", tags=["navigate"])
    app.include_router(voice_router, prefix="/api/v1", tags=["voice"])
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

    web_dir = Path(__file__).resolve().parents[1] / "frontend" / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
