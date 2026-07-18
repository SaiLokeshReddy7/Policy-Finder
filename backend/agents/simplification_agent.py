"""Language Simplification Agent: rewrites the eligibility reasoning + benefits
into plain, localized language. Primary model is an open-source Hugging Face
instruction-tuned model (via HF Inference API); falls back to a small Claude
model if no HF token is configured or the HF call fails."""
from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.core.config import get_model_settings
from backend.llm.anthropic_client import complete
from backend.models.schemas import EligibilityVerdict
from backend.tools.hf_inference_tool import simplify_text

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}

FALLBACK_SYSTEM_PROMPT = (
    "You rewrite government welfare scheme information in simple, plain {language} "
    "that a first-time applicant with no legal/policy background can understand in "
    "under 90 seconds. Keep it to 3-4 short sentences. Be encouraging but accurate; "
    "do not invent eligibility rules."
)

RELEVANT_VERDICTS = {
    EligibilityVerdict.likely_eligible,
    EligibilityVerdict.possibly_eligible,
    EligibilityVerdict.needs_more_info,
}


def _build_prompt(language_name: str, name: str, benefits: str, reason: str, documents: list[str]) -> str:
    docs = ", ".join(documents) if documents else "standard KYC documents"
    return (
        f"Rewrite the following government welfare scheme information in simple, plain "
        f"{language_name} for a citizen with no policy background. Keep it to 3-4 short "
        f"sentences covering: what the scheme gives them, why they likely qualify (or what's "
        f"unclear), and the main documents needed. Be warm and encouraging but accurate.\n\n"
        f"Scheme: {name}\nBenefits: {benefits}\nEligibility reasoning: {reason}\n"
        f"Documents: {docs}\n\nSimplified explanation in {language_name}:"
    )


def run_simplification_agent(state: AgentState) -> AgentState:
    candidates = {c.scheme_id: c for c in state.get("candidates", [])}
    eligibility_results = state.get("eligibility_results", [])
    guidance = state.get("document_guidance", {})
    language = state.get("language", "en")
    language_name = LANGUAGE_NAMES.get(language, "English")
    settings = get_model_settings().anthropic

    explanations: dict[str, str] = {}
    for r in eligibility_results:
        if r.verdict not in RELEVANT_VERDICTS:
            continue
        candidate = candidates.get(r.scheme_id)
        if candidate is None:
            continue
        doc = guidance.get(r.scheme_id)
        documents = doc.required_documents if doc else candidate.required_documents
        prompt = _build_prompt(language_name, candidate.name, candidate.benefits, r.reason, documents)

        text = None
        try:
            text = simplify_text(prompt)
        except Exception:
            logger.warning("HF simplification call raised for scheme_id=%s", r.scheme_id, exc_info=True)

        if not text:
            try:
                text = complete(
                    model=settings.models.simplification_fallback,
                    system_prompt=FALLBACK_SYSTEM_PROMPT.format(language=language_name),
                    user_prompt=prompt,
                    max_tokens=250,
                )
            except Exception:
                logger.warning("Claude simplification fallback failed for scheme_id=%s", r.scheme_id, exc_info=True)
                text = r.reason

        explanations[r.scheme_id] = text

    state["simplified_explanations"] = explanations
    return state
