"""Thin HTTP client wrapping the FastAPI backend for the Streamlit app."""
from __future__ import annotations

import os

import httpx

BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")


def _headers() -> dict:
    headers = {}
    if BACKEND_API_KEY:
        headers["X-API-Key"] = BACKEND_API_KEY
    return headers


class BackendError(RuntimeError):
    pass


def get_health() -> dict:
    response = httpx.get(f"{BACKEND_BASE_URL}/api/v1/health", headers=_headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def list_schemes() -> list[dict]:
    response = httpx.get(f"{BACKEND_BASE_URL}/api/v1/schemes", headers=_headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def navigate(profile: dict, free_text_context: str | None) -> dict:
    payload = {"profile": profile, "language": "en", "free_text_context": free_text_context}
    try:
        response = httpx.post(
            f"{BACKEND_BASE_URL}/api/v1/navigate", json=payload, headers=_headers(), timeout=90
        )
    except httpx.ConnectError as exc:
        raise BackendError(
            f"Could not reach the backend at {BACKEND_BASE_URL}. Is it running "
            f"(uvicorn backend.main:app --reload)?"
        ) from exc
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise BackendError(str(detail))
    return response.json()


def chat(message: str, history: list[dict], profile_summary: str, recommendations: list[dict]) -> str:
    """Asks a follow-up question grounded in the last set of recommendations."""
    payload = {
        "message": message,
        "history": history,
        "profile_summary": profile_summary,
        "recommendations": recommendations,
    }
    try:
        response = httpx.post(
            f"{BACKEND_BASE_URL}/api/v1/chat", json=payload, headers=_headers(), timeout=60
        )
    except httpx.ConnectError as exc:
        raise BackendError(f"Could not reach the backend at {BACKEND_BASE_URL}.") from exc
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise BackendError(str(detail))
    return response.json()["reply"]


def transcribe_voice(audio_bytes: bytes, content_type: str = "audio/wav") -> dict:
    """Uploads a voice recording and gets back {transcript, suggested_profile}."""
    try:
        response = httpx.post(
            f"{BACKEND_BASE_URL}/api/v1/voice/transcribe",
            files={"file": ("recording.wav", audio_bytes, content_type or "audio/wav")},
            headers=_headers(),
            timeout=60,
        )
    except httpx.ConnectError as exc:
        raise BackendError(f"Could not reach the backend at {BACKEND_BASE_URL}.") from exc
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise BackendError(str(detail))
    return response.json()


def synthesize_speech(text: str, language: str) -> tuple[bytes, str]:
    """Returns (audio_bytes, content_type) for spoken playback of `text`."""
    try:
        response = httpx.post(
            f"{BACKEND_BASE_URL}/api/v1/voice/speak",
            json={"text": text, "language": language},
            headers=_headers(),
            timeout=60,
        )
    except httpx.ConnectError as exc:
        raise BackendError(f"Could not reach the backend at {BACKEND_BASE_URL}.") from exc
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise BackendError(str(detail))
    return response.content, response.headers.get("content-type", "audio/flac")
