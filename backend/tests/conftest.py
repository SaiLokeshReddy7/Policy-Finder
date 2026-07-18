"""Shared pytest fixtures.

Design goals:
- No test ever makes a real network call (no Anthropic, no HF Inference API,
  no DuckDuckGo, no downloading a real sentence-transformers model).
- The scheme knowledge base / vector index is built once per test session
  against a throwaway temp directory using fake, cheap embeddings, so tests
  never touch (or depend on) the real data/vectorstore/ on disk.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("HF_API_TOKEN", "")
os.environ.setdefault("BACKEND_API_KEY", "")

from backend.core import config as config_module  # noqa: E402


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic, cheap stand-in for the real HF embedding model."""
    vectors = np.zeros((len(texts), 32), dtype="float32")
    for i, text in enumerate(texts):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vectors[i] = rng.random(32).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _fake_embed_query(text: str) -> np.ndarray:
    return _fake_embed_texts([text])[0]


@pytest.fixture(scope="session", autouse=True)
def _use_temp_vector_store(tmp_path_factory):
    """Point the scheme service at a throwaway vector store directory so
    tests never read/write the real data/vectorstore/ on disk."""
    settings = config_module.get_model_settings()
    temp_dir = tmp_path_factory.mktemp("vectorstore")
    settings.app.vector_store_path = str(temp_dir)
    yield


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    import backend.services.scheme_service as scheme_service_module

    monkeypatch.setattr(scheme_service_module, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(scheme_service_module, "embed_query", _fake_embed_query)
    yield


@pytest.fixture
def mock_llm_calls(monkeypatch):
    """Stubs out every Claude / HF Inference call used by the six agents,
    keyed off the scheme_id values present in each agent's own prompt so
    the fakes stay correct regardless of which candidates a test provides."""
    import re

    import backend.agents.document_agent as document_agent
    import backend.agents.eligibility_agent as eligibility_agent
    import backend.agents.intake_agent as intake_agent
    import backend.agents.simplification_agent as simplification_agent

    def fake_complete(model, system_prompt, user_prompt, max_tokens=1024):
        return "45-year-old test citizen profile summary for eligibility reasoning."

    def fake_eligibility_json(model, system_prompt, user_prompt, max_tokens=2048):
        ids = re.findall(r"'scheme_id':\s*'([^']+)'", user_prompt)
        return [
            {"scheme_id": sid, "verdict": "Likely Eligible", "reason": "Matches test profile.", "confidence": 0.9}
            for sid in dict.fromkeys(ids)
        ]

    def fake_document_json(model, system_prompt, user_prompt, max_tokens=2048):
        ids = re.findall(r"'scheme_id':\s*'([^']+)'", user_prompt)
        return [
            {
                "scheme_id": sid,
                "required_documents": ["Aadhaar card", "Bank account details"],
                "application_steps": ["Visit the official portal", "Submit the application with documents"],
                "common_blockers": ["Missing income certificate"],
            }
            for sid in dict.fromkeys(ids)
        ]

    monkeypatch.setattr(intake_agent, "complete", fake_complete)
    monkeypatch.setattr(eligibility_agent, "complete_json", fake_eligibility_json)
    monkeypatch.setattr(document_agent, "complete_json", fake_document_json)
    monkeypatch.setattr(simplification_agent, "simplify_text", lambda prompt: "Simplified test explanation.")
    monkeypatch.setattr(simplification_agent, "complete", fake_complete)
    yield


@pytest.fixture
def sample_scheme() -> dict:
    return {
        "id": "pm-kisan",
        "name": "Pradhan Mantri Kisan Samman Nidhi",
        "short_name": "PM-KISAN",
        "category": "Agriculture & Farmers",
        "description": "Income support for landholding farmer families.",
        "benefits": "INR 6,000/year direct benefit transfer.",
        "eligibility": {"occupation": ["farmer"], "land_holding_required": True},
        "required_documents": ["Aadhaar card", "Land ownership records"],
        "how_to_apply": "Register at pmkisan.gov.in.",
        "source_name": "PM-KISAN Official Portal",
        "source_url": "https://pmkisan.gov.in",
    }
