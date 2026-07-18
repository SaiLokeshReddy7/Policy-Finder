from __future__ import annotations

from fastapi.testclient import TestClient

import backend.api.routes_chat as routes_chat
from backend.main import app

client = TestClient(app)

SAMPLE_RECOMMENDATION = {
    "scheme_id": "pm-kisan",
    "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
    "category": "Agriculture & Farmers",
    "verdict": "Likely Eligible",
    "reason": "Owns cultivable farmland.",
    "confidence": 0.9,
    "benefits": "INR 6,000 per year in three installments.",
    "required_documents": ["Aadhaar card", "Land records"],
    "application_steps": ["Register at pmkisan.gov.in"],
    "common_blockers": [],
    "simplified_explanation": "You get 6000 rupees a year as a farmer.",
    "source": {"name": "PM-KISAN Portal", "url": "https://pmkisan.gov.in", "origin": "knowledge_base"},
}


def test_chat_grounds_reply_in_recommendation_context(monkeypatch):
    captured = {}

    def fake_chat_complete(model, system_prompt, messages, max_tokens=1024):
        captured["system_prompt"] = system_prompt
        captured["messages"] = messages
        return "PM-KISAN pays INR 6,000 per year. See https://pmkisan.gov.in."

    monkeypatch.setattr(routes_chat, "chat_complete", fake_chat_complete)

    payload = {
        "message": "How much does PM-KISAN pay?",
        "history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! Ask me about your schemes."},
        ],
        "profile_summary": "A 45-year-old farmer from Bihar.",
        "recommendations": [SAMPLE_RECOMMENDATION],
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    assert "6,000" in response.json()["reply"]

    # scheme context and profile summary must be in the system prompt
    assert "Pradhan Mantri Kisan Samman Nidhi" in captured["system_prompt"]
    assert "45-year-old farmer" in captured["system_prompt"]
    # history + new message forwarded in order, ending with the user's question
    assert captured["messages"][-1] == {"role": "user", "content": "How much does PM-KISAN pay?"}
    assert len(captured["messages"]) == 3


def test_chat_rejects_empty_message():
    response = client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_returns_502_when_llm_fails(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(routes_chat, "chat_complete", boom)
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 502
