"""Retrieval Agent: hybrid search over the local scheme knowledge base
(kb_search tool) with a live web_search tool fallback for coverage gaps."""
from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.core.config import get_model_settings
from backend.models.schemas import SchemeCandidate, SchemeSource
from backend.tools.kb_search_tool import kb_search
from backend.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

MIN_CANDIDATES_BEFORE_WEB_SEARCH = 3


def _build_query(state: AgentState) -> str:
    profile = state["profile"]
    parts = [state.get("profile_summary", "")]
    if profile.is_farmer:
        parts.append("farmer agriculture scheme")
    if profile.is_student:
        parts.append("student scholarship")
    if profile.is_disabled:
        parts.append("disability assistance scheme")
    if profile.is_bpl:
        parts.append("below poverty line welfare scheme")
    return " ".join(p for p in parts if p)


def run_retrieval_agent(state: AgentState) -> AgentState:
    settings = get_model_settings().app
    query = _build_query(state)

    candidates: list[SchemeCandidate] = []
    seen_ids: set[str] = set()

    kb_hits = kb_search.invoke({"query": query, "top_k": settings.max_schemes_returned})
    for scheme in kb_hits:
        if scheme["id"] in seen_ids:
            continue
        seen_ids.add(scheme["id"])
        candidates.append(
            SchemeCandidate(
                scheme_id=scheme["id"],
                name=scheme["name"],
                category=scheme.get("category", ""),
                description=scheme.get("description", ""),
                benefits=scheme.get("benefits", ""),
                eligibility_rules=scheme.get("eligibility", {}) or {},
                required_documents=scheme.get("required_documents", []),
                how_to_apply=scheme.get("how_to_apply", ""),
                source=SchemeSource(
                    name=scheme.get("source_name", ""),
                    url=scheme.get("source_url", ""),
                    origin="knowledge_base",
                ),
            )
        )

    warnings = list(state.get("warnings", []))
    if len(candidates) < MIN_CANDIDATES_BEFORE_WEB_SEARCH:
        profile = state["profile"]
        web_query = f"{query} India government welfare scheme {profile.state}".strip()
        try:
            web_results = web_search_tool.invoke({"query": web_query, "max_results": 5})
        except Exception:
            logger.warning("Web search tool failed", exc_info=True)
            web_results = []
            warnings.append("Live web search was unavailable; results are limited to the local knowledge base.")

        for result in web_results:
            url = result.get("url", "")
            if not url or url in seen_ids:
                continue
            seen_ids.add(url)
            candidates.append(
                SchemeCandidate(
                    scheme_id=f"web:{url}",
                    name=result.get("title") or "Untitled result",
                    category="Web result",
                    description=result.get("snippet", ""),
                    benefits="",
                    eligibility_rules={},
                    required_documents=[],
                    how_to_apply="",
                    source=SchemeSource(name=result.get("title", ""), url=url, origin="web_search"),
                )
            )

    state["candidates"] = candidates
    state["warnings"] = warnings
    return state
