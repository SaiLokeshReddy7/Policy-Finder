"""Thin Anthropic (Claude) client used by the core reasoning agents:
Intake, Eligibility Reasoning, Document Guidance, and the Claude fallback
path for Language Simplification."""
from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_model_settings

logger = logging.getLogger(__name__)


class AnthropicClientError(RuntimeError):
    pass


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        settings = get_model_settings().anthropic
        if not settings.api_key:
            raise AnthropicClientError(
                "ANTHROPIC_API_KEY is not configured. Set it in .env (referenced by "
                "config/models.yaml as ${ANTHROPIC_API_KEY}) before calling any Claude agent."
            )
        _client = Anthropic(api_key=settings.api_key, base_url=settings.base_url, max_retries=0)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def complete(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Single-turn Claude completion; returns the concatenated text content.

    Deliberately does not send `temperature` -- current Claude Sonnet 5 rejects
    it outright ("`temperature` is deprecated for this model"), confirmed live
    while building this project. Omitting it works across all Claude models
    used here and just falls back to each model's own default sampling.
    """
    settings = get_model_settings().anthropic
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=settings.timeout_seconds,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
def chat_complete(
    model: str,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    """Multi-turn Claude completion for the follow-up chat endpoint.

    `messages` is a list of {"role": "user"|"assistant", "content": str} turns,
    ending with the newest user message. Same no-`temperature` rule as complete().
    """
    settings = get_model_settings().anthropic
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        timeout=settings.timeout_seconds,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def complete_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1536,
) -> Any:
    """Calls Claude and parses the response as JSON. Callers' system prompts
    must instruct the model to return a single JSON value and nothing else."""
    raw = complete(model, system_prompt, user_prompt, max_tokens=max_tokens)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude JSON response: %s\nRaw response: %s", exc, raw)
        raise AnthropicClientError(f"Claude did not return valid JSON: {exc}") from exc
