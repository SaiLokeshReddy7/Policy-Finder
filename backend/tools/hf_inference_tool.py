"""Client for the Hugging Face Inference Providers router, used by the
Language Simplification Agent to run an open-source instruction-tuned model.

HF retired the old classic serverless "api-inference.huggingface.co" API;
chat/instruct models are now served through a unified, OpenAI-compatible
chat-completions endpoint at router.huggingface.co, which auto-selects
whichever inference provider (hf-inference, together, featherless-ai, etc.)
actually serves the requested model -- verified live against
Qwen/Qwen2.5-7B-Instruct while building this feature. Which provider ends up
serving a given model, and whether that requires provider credits beyond
HF's small free monthly allowance, is outside this app's control.

Deliberately fails soft: if HF_API_TOKEN isn't configured or the call errors
out (rate limit, no provider available, network), callers get None back and
fall back to the Claude simplification_fallback model instead of crashing the
whole navigator pipeline.
"""
from __future__ import annotations

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_model_settings

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class HFInferenceError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
def _call_hf_chat(model: str, prompt: str, api_key: str, base_url: str, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}{CHAT_COMPLETIONS_PATH}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HFInferenceError(f"Unexpected HF chat-completions response shape: {data!r}") from exc


def simplify_text(prompt: str) -> str | None:
    """Calls the configured open-source HF model to simplify/localize text.
    Returns None if HF isn't configured or the call fails."""
    settings = get_model_settings().huggingface
    if not settings.api_key:
        logger.debug("HF_API_TOKEN not configured; skipping HF simplification model")
        return None
    try:
        return _call_hf_chat(
            model=settings.models.simplification,
            prompt=prompt,
            api_key=settings.api_key,
            base_url=settings.inference_api_url,
            timeout=settings.timeout_seconds,
        )
    except Exception:
        logger.warning("HF Inference call failed; falling back to Claude", exc_info=True)
        return None
