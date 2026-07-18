"""LangGraph/LangChain tool wrapping the local scheme knowledge-base search."""
from __future__ import annotations

from langchain_core.tools import tool

from backend.services.scheme_service import get_scheme_service


@tool("kb_search")
def kb_search(query: str, top_k: int = 8) -> list[dict]:
    """Search the local Indian welfare-scheme knowledge base for schemes
    matching a natural-language description of a citizen's situation
    (e.g. "small farmer low income Bihar" or "widow pension BPL"). Returns a
    list of raw scheme records (id, name, category, description, benefits,
    eligibility, required_documents, source_name, source_url)."""
    service = get_scheme_service()
    hits = service.search(query, top_k=top_k)
    return [scheme for scheme, _score in hits]
