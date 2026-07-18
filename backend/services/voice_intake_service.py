"""Best-effort extraction of structured CitizenProfile fields from a spoken
transcript, so the voice input feature can pre-fill the profile form instead
of only dropping the transcript into the free-text box. This is a UI
convenience step outside the main LangGraph pipeline -- it never blocks or
fails the transcription response if it can't confidently extract anything."""
from __future__ import annotations

import logging

from backend.core.config import get_model_settings
from backend.llm.anthropic_client import complete_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract structured household-profile fields from a citizen's
spoken transcript (which may be in English, Hindi, or Telugu, and may be
imperfectly transcribed). Return a single JSON object containing ONLY the
fields you can confidently determine from the transcript -- omit any field
you're not reasonably sure about. Never guess a specific number if none was
stated or clearly implied.

Allowed fields and formats (all optional):
- age: integer
- gender: one of "male", "female", "other"
- annual_income: integer, household annual income in INR (convert monthly to
  yearly, lakhs to INR, etc. if stated that way)
- occupation: short free-text string
- state: an Indian state/UT name
- category: one of "General", "SC", "ST", "OBC", "Minority"
- is_student: boolean
- is_disabled: boolean
- is_farmer: boolean
- is_bpl: boolean
- family_size: integer

Return ONLY the JSON object, no prose, no markdown fences. If nothing can be
confidently extracted, return {}."""


def extract_profile_fields(transcript: str) -> dict:
    if not transcript or not transcript.strip():
        return {}

    settings = get_model_settings().anthropic
    try:
        result = complete_json(
            model=settings.models.intake,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Transcript: {transcript}",
            max_tokens=400,
        )
        return result if isinstance(result, dict) else {}
    except Exception:
        logger.warning("Voice profile field extraction failed; returning transcript only", exc_info=True)
        return {}
