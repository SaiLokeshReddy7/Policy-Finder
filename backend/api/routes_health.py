"""Liveness/readiness endpoint for local dev, Docker healthchecks, and k8s probes."""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import get_model_settings
from backend.models.schemas import HealthStatus
from backend.services.scheme_service import get_scheme_service

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    settings = get_model_settings()
    service = get_scheme_service()
    return HealthStatus(
        status="ok",
        vector_store_loaded=service.is_loaded,
        schemes_indexed=service.scheme_count,
        anthropic_configured=bool(settings.anthropic.api_key),
        huggingface_configured=bool(settings.huggingface.api_key),
        search_provider=settings.search.provider,
    )
