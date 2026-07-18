"""Free internet search used by the Retrieval Agent to find schemes that
aren't in the local knowledge base (e.g. state-specific schemes).

Defaults to DuckDuckGo (via the `ddgs` package), which needs zero API keys.
Set search.provider in config/models.yaml to "tavily" or "serpapi" (plus the
corresponding API key) for higher-quality results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_model_settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _duckduckgo_search(query: str, max_results: int) -> list[WebSearchResult]:
    from ddgs import DDGS

    results: list[WebSearchResult] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("url", "")),
                    snippet=r.get("body", ""),
                )
            )
    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _tavily_search(query: str, max_results: int, api_key: str) -> list[WebSearchResult]:
    response = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return [
        WebSearchResult(title=item.get("title", ""), url=item.get("url", ""), snippet=item.get("content", ""))
        for item in data.get("results", [])
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _serpapi_search(query: str, max_results: int, api_key: str) -> list[WebSearchResult]:
    response = httpx.get(
        "https://serpapi.com/search",
        params={"q": query, "api_key": api_key, "num": max_results, "engine": "google"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    organic = data.get("organic_results", [])[:max_results]
    return [
        WebSearchResult(title=item.get("title", ""), url=item.get("link", ""), snippet=item.get("snippet", ""))
        for item in organic
    ]


def web_search(query: str, max_results: int | None = None) -> list[WebSearchResult]:
    """Provider-agnostic web search dispatcher, selected via config/models.yaml."""
    settings = get_model_settings().search
    max_results = max_results or settings.max_results
    try:
        if settings.provider == "tavily" and settings.tavily_api_key:
            return _tavily_search(query, max_results, settings.tavily_api_key)
        if settings.provider == "serpapi" and settings.serpapi_api_key:
            return _serpapi_search(query, max_results, settings.serpapi_api_key)
        return _duckduckgo_search(query, max_results)
    except Exception:
        logger.exception("Web search failed for query=%r", query)
        return []


@tool("web_search")
def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search the public internet for information about government welfare
    schemes not present in the local knowledge base. Returns a list of
    {title, url, snippet} dicts, always with a real source URL for
    traceability."""
    results = web_search(query, max_results=max_results)
    return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
