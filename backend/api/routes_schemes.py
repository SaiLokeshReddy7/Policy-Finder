"""Read-only listing of the local scheme knowledge base (useful for the
Streamlit UI's "browse all schemes" view and for debugging)."""
from __future__ import annotations

from fastapi import APIRouter

from backend.models.schemas import SchemeSource, SchemeSummary
from backend.services.scheme_service import get_scheme_service

router = APIRouter()


@router.get("/schemes", response_model=list[SchemeSummary])
def list_schemes() -> list[SchemeSummary]:
    service = get_scheme_service()
    return [
        SchemeSummary(
            scheme_id=s["id"],
            name=s["name"],
            category=s.get("category", ""),
            description=s.get("description", ""),
            source=SchemeSource(
                name=s.get("source_name", ""), url=s.get("source_url", ""), origin="knowledge_base"
            ),
        )
        for s in service.list_schemes()
    ]
