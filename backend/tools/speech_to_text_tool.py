"""Speech-to-text via the Hugging Face Inference Providers router
(open-source Whisper, served free by HF's own "hf-inference" provider --
verified live for openai/whisper-large-v3-turbo while building this feature).

Powers the "speak your situation" voice input feature. There is no Claude
fallback here -- the Anthropic API doesn't accept raw audio, so this is one
of the clearest cases in the project where the open-source HF model is doing
work Claude structurally can't. If HF isn't configured, the feature is simply
unavailable and the caller (routes_voice.py) returns a clear error.
"""
from __future__ import annotations

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_model_settings

logger = logging.getLogger(__name__)

HF_INFERENCE_PROVIDER_PATH = "/hf-inference/models"


class SpeechToTextError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
def _call_hf_asr(model: str, audio_bytes: bytes, content_type: str, api_key: str, base_url: str, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}{HF_INFERENCE_PROVIDER_PATH}/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": content_type or "audio/wav",
    }
    response = httpx.post(url, headers=headers, content=audio_bytes, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = data.get("text") if isinstance(data, dict) else None
    if not text:
        raise SpeechToTextError(f"Unexpected HF ASR response shape: {data!r}")
    return text.strip()


def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> str | None:
    """Transcribes recorded speech (English/Hindi/Telugu, auto-detected) to
    text. Returns None if HF isn't configured or the call fails, rather than
    raising, so the route can turn that into a clean user-facing error."""
    settings = get_model_settings().huggingface
    if not settings.api_key:
        logger.info("HF_API_TOKEN not configured; speech-to-text is unavailable")
        return None
    if not audio_bytes:
        return None
    try:
        return _call_hf_asr(
            model=settings.models.speech_to_text,
            audio_bytes=audio_bytes,
            content_type=content_type,
            api_key=settings.api_key,
            base_url=settings.inference_api_url,
            timeout=settings.timeout_seconds,
        )
    except Exception:
        logger.warning("HF speech-to-text call failed", exc_info=True)
        return None
