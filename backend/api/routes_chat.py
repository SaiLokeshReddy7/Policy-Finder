"""Follow-up chat endpoint: answers the citizen's questions about the
recommendations they just received, grounded in that result set."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.core.config import get_model_settings
from backend.core.rate_limit import limiter
from backend.llm.anthropic_client import chat_complete
from backend.models.schemas import ChatRequest, ChatResponse, SchemeRecommendation

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_HISTORY_TURNS = 12

SYSTEM_PROMPT = """You are the follow-up assistant of a public welfare-scheme navigator
for Indian citizens. The user has just received the scheme recommendations shown below
and is asking follow-up questions about them.

Answer BRIEFLY and directly:
- Start with the direct answer in the FIRST sentence. No preamble.
- Keep the whole reply under ~80 words. Use at most 3 short bullet points, and only when
  a list genuinely helps. Do NOT add section headings like "Why these first" or
  "What about the others".
- Only mention the scheme(s) the question is actually about. Do not recap every scheme.
- Expand into a fuller, step-by-step answer ONLY if the user explicitly asks for more
  detail ("explain more", "full steps", "everything about X").

Accuracy rules:
- Ground every answer in the recommendation context provided (verdict, reason, benefits,
  documents, steps, source). Never invent scheme names, amounts, or eligibility rules.
- If the answer isn't in the context and you're unsure, say so in one line and point to
  the scheme's official source link instead of guessing.
- Use plain words a non-expert can follow. If asked something off-topic, steer back in
  one sentence."""


def _build_context(profile_summary: str, recommendations: list[SchemeRecommendation]) -> str:
    lines = []
    if profile_summary:
        lines.append(f"Citizen profile summary: {profile_summary}")
    if recommendations:
        lines.append("Recommendations shown to the user:")
        for rec in recommendations:
            lines.append(
                f"- {rec.scheme_name} [{rec.scheme_id}] ({rec.category}): "
                f"{rec.verdict.value}, confidence {rec.confidence:.0%}. "
                f"Reason: {rec.reason} Benefits: {rec.benefits} "
                f"Documents: {', '.join(rec.required_documents) or 'n/a'}. "
                f"Steps: {' | '.join(rec.application_steps) or 'n/a'}. "
                f"Source: {rec.source.name} ({rec.source.url})"
            )
    else:
        lines.append("No recommendations context was provided.")
    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    settings = get_model_settings().anthropic
    system_prompt = (
        SYSTEM_PROMPT
        + "\n\n--- CONTEXT ---\n"
        + _build_context(payload.profile_summary, payload.recommendations)
    )
    messages = [
        {"role": turn.role, "content": turn.content}
        for turn in payload.history[-MAX_HISTORY_TURNS:]
    ]
    messages.append({"role": "user", "content": payload.message})
    try:
        reply = chat_complete(
            model=settings.models.chat,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(
            status_code=502,
            detail="The assistant could not answer right now. Please try again.",
        ) from exc
    return ChatResponse(reply=reply)
