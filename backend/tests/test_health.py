from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["schemes_indexed"] > 0
    assert body["anthropic_configured"] is True
    assert body["search_provider"] == "duckduckgo"


def test_list_schemes_returns_seed_data():
    response = client.get("/api/v1/schemes")
    assert response.status_code == 200
    schemes = response.json()
    assert len(schemes) > 0
    ids = {s["scheme_id"] for s in schemes}
    assert "pm-kisan" in ids
    assert all(s["source"]["url"].startswith("http") for s in schemes)
