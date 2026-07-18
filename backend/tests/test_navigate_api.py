from __future__ import annotations

from fastapi.testclient import TestClient

import backend.agents.retrieval_agent as retrieval_agent
from backend.main import app

client = TestClient(app)


class _FakeTool:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, args: dict):
        return self._fn(**args)


def _patch_retrieval_tools(monkeypatch, sample_scheme: dict):
    def fake_kb_search(query: str, top_k: int = 8):
        return [sample_scheme]

    def fake_web_search(query: str, max_results: int = 5):
        return []

    monkeypatch.setattr(retrieval_agent, "kb_search", _FakeTool(fake_kb_search))
    monkeypatch.setattr(retrieval_agent, "web_search_tool", _FakeTool(fake_web_search))


def test_navigate_returns_traceable_recommendations(monkeypatch, mock_llm_calls, sample_scheme):
    _patch_retrieval_tools(monkeypatch, sample_scheme)

    payload = {
        "profile": {
            "age": 45,
            "gender": "male",
            "annual_income": 120000,
            "occupation": "farmer",
            "state": "Bihar",
            "category": "General",
            "is_farmer": True,
        },
        "language": "en",
        "free_text_context": "I own two acres of farmland and have no other income.",
    }

    response = client.post("/api/v1/navigate", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["recommendations"], "expected at least one recommendation"
    rec = body["recommendations"][0]
    assert rec["scheme_id"] == sample_scheme["id"]
    assert rec["verdict"] == "Likely Eligible"
    assert rec["source"]["url"] == sample_scheme["source_url"]
    assert rec["simplified_explanation"]
    assert "profile_summary" in body


def test_navigate_rejects_invalid_profile():
    payload = {"profile": {"age": -5, "gender": "male", "annual_income": 1000}}
    response = client.post("/api/v1/navigate", json=payload)
    assert response.status_code == 422


def test_navigate_returns_502_when_graph_fails(monkeypatch, sample_scheme):
    # The Intake Agent already falls back gracefully on LLM failure, so to
    # exercise the API's 502 path we simulate a harder failure with no
    # fallback: a malformed KB record missing required fields.
    monkeypatch.setattr(
        retrieval_agent,
        "kb_search",
        _FakeTool(lambda query, top_k=8: [{"id": "broken"}]),
    )
    monkeypatch.setattr(retrieval_agent, "web_search_tool", _FakeTool(lambda query, max_results=5: []))

    payload = {
        "profile": {
            "age": 45,
            "gender": "male",
            "annual_income": 120000,
            "occupation": "farmer",
            "state": "Bihar",
        },
        "language": "en",
    }
    response = client.post("/api/v1/navigate", json=payload)
    assert response.status_code == 502
