from __future__ import annotations

from fastapi.testclient import TestClient

import backend.api.routes_voice as routes_voice
import backend.services.voice_intake_service as voice_intake_service
from backend.main import app

client = TestClient(app)


def test_voice_transcribe_success(monkeypatch):
    monkeypatch.setattr(routes_voice, "transcribe_audio", lambda audio_bytes, content_type=None: "I am a 45 year old farmer in Bihar.")
    monkeypatch.setattr(
        routes_voice,
        "extract_profile_fields",
        lambda transcript: {"age": 45, "occupation": "farmer", "state": "Bihar", "is_farmer": True},
    )

    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("recording.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "I am a 45 year old farmer in Bihar."
    assert body["suggested_profile"]["age"] == 45
    assert body["suggested_profile"]["is_farmer"] is True


def test_voice_transcribe_returns_503_when_hf_unconfigured(monkeypatch):
    monkeypatch.setattr(routes_voice, "transcribe_audio", lambda audio_bytes, content_type=None: None)

    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("recording.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert response.status_code == 503
    assert "HF_API_TOKEN" in response.json()["detail"]


def test_voice_transcribe_rejects_empty_audio():
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("recording.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_voice_transcribe_rejects_oversized_audio(monkeypatch):
    monkeypatch.setattr(routes_voice, "MAX_AUDIO_BYTES", 10)
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("recording.wav", b"x" * 100, "audio/wav")},
    )
    assert response.status_code == 413


def test_voice_speak_success(monkeypatch):
    monkeypatch.setattr(routes_voice, "synthesize_speech", lambda text, language: (b"fake-audio-bytes", "audio/flac"))

    response = client.post("/api/v1/voice/speak", json={"text": "You are likely eligible.", "language": "en"})
    assert response.status_code == 200
    assert response.content == b"fake-audio-bytes"
    assert response.headers["content-type"] == "audio/flac"


def test_voice_speak_returns_503_when_unavailable(monkeypatch):
    monkeypatch.setattr(routes_voice, "synthesize_speech", lambda text, language: None)

    response = client.post("/api/v1/voice/speak", json={"text": "Hello", "language": "fr"})
    assert response.status_code == 503


def test_extract_profile_fields_returns_empty_dict_on_llm_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("Anthropic API unreachable")

    monkeypatch.setattr(voice_intake_service, "complete_json", _boom)
    assert voice_intake_service.extract_profile_fields("some transcript") == {}


def test_extract_profile_fields_returns_empty_dict_for_blank_transcript():
    assert voice_intake_service.extract_profile_fields("   ") == {}
