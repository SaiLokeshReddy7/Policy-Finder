"""Intake Agent: normalizes the structured form + optional free-text context
into a natural-language profile summary used by later agents' prompts."""
from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.core.config import get_model_settings
from backend.llm.anthropic_client import complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Intake Agent of a public welfare-scheme navigator.
You receive a citizen's structured household profile plus optional free-text context
they typed in their own words. Write ONE concise paragraph (3-5 sentences) in English
that summarizes the household for downstream eligibility reasoning: age, gender,
occupation, approximate income, location, family size, category, and student/
disability/farmer/BPL status. Fold in anything notable from the free-text context
(e.g. recent job loss, pregnancy, migrant status). Do not invent facts that are not
present in the input. Output only the paragraph, no preamble, no headers."""


def run_intake_agent(state: AgentState) -> AgentState:
    profile = state["profile"]
    settings = get_model_settings().anthropic
    user_prompt = (
        f"Structured profile (JSON): {profile.model_dump_json()}\n"
        f"Free-text context: {state.get('free_text_context') or 'None provided'}"
    )
    try:
        summary = complete(
            model=settings.models.intake,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=300,
        )
    except Exception:
        logger.warning("Intake agent LLM call failed; falling back to a templated summary", exc_info=True)
        summary = (
            f"{profile.age}-year-old {profile.gender.value} in {profile.state or 'India'}, "
            f"occupation: {profile.occupation or 'not specified'}, category: {profile.category.value}, "
            f"household annual income approx INR {profile.annual_income}, family size {profile.family_size}."
        )
    state["profile_summary"] = summary
    return state
