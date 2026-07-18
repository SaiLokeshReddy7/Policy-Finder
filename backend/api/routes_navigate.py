"""The core endpoint: runs the LangGraph multi-agent pipeline for a citizen
profile and returns ranked, traceable scheme recommendations."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from backend.agents.graph import get_navigator_graph
from backend.core.rate_limit import limiter
from backend.models.schemas import NavigateRequest, NavigateResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/navigate", response_model=NavigateResponse)
@limiter.limit("10/minute")
def navigate(request: Request, payload: NavigateRequest) -> NavigateResponse:
    graph = get_navigator_graph()
    initial_state = {
        "profile": payload.profile,
        "free_text_context": payload.free_text_context,
        "language": payload.language,
        "warnings": [],
    }
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Navigator graph execution failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "The navigator agents failed to complete the request. Check that "
                "ANTHROPIC_API_KEY is set correctly in .env / config/models.yaml."
            ),
        ) from exc

    return NavigateResponse(
        profile_summary=final_state.get("profile_summary", ""),
        language=payload.language,
        recommendations=final_state.get("recommendations", []),
        generated_at=datetime.now(timezone.utc).isoformat(),
        warnings=final_state.get("warnings", []),
    )
