"""Document Guidance Agent: produces a required-documents checklist,
application steps, and common blockers for each relevant scheme."""
from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.core.config import get_model_settings
from backend.llm.anthropic_client import complete_json
from backend.models.schemas import DocumentGuidance, EligibilityVerdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Document Guidance Agent of a public welfare-scheme
navigator. For each given scheme (with its known required documents, if any, and a
short description/how-to-apply note), produce practical guidance for an Indian citizen
applying for it. Return ONLY a JSON array where each object has: scheme_id,
required_documents (list of strings; use the given ones as the base and add any obviously
missing standard KYC documents), application_steps (ordered list of 3-6 short steps),
common_blockers (list of 2-4 short, concrete reasons applications for this type of scheme
typically get rejected or delayed). No prose, no markdown code fences, no extra keys."""

RELEVANT_VERDICTS = {
    EligibilityVerdict.likely_eligible,
    EligibilityVerdict.possibly_eligible,
    EligibilityVerdict.needs_more_info,
}


def run_document_agent(state: AgentState) -> AgentState:
    candidates = {c.scheme_id: c for c in state.get("candidates", [])}
    eligibility_results = state.get("eligibility_results", [])
    settings = get_model_settings().anthropic

    relevant = [r for r in eligibility_results if r.verdict in RELEVANT_VERDICTS]
    guidance: dict[str, DocumentGuidance] = {}

    if not relevant:
        state["document_guidance"] = guidance
        return state

    payload = []
    for r in relevant:
        c = candidates.get(r.scheme_id)
        if c is None:
            continue
        payload.append(
            {
                "scheme_id": c.scheme_id,
                "name": c.name,
                "known_required_documents": c.required_documents,
                "how_to_apply": c.how_to_apply,
                "description": c.description,
            }
        )

    try:
        raw = complete_json(
            model=settings.models.document_guidance,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Schemes (JSON): {payload}",
            max_tokens=2048,
        )
        for item in raw:
            scheme_id = item.get("scheme_id")
            if scheme_id not in candidates:
                continue
            guidance[scheme_id] = DocumentGuidance(
                scheme_id=scheme_id,
                required_documents=item.get("required_documents", []),
                application_steps=item.get("application_steps", []),
                common_blockers=item.get("common_blockers", []),
            )
    except Exception:
        logger.warning("Document guidance agent LLM call failed; falling back to KB documents only", exc_info=True)
        for item in payload:
            guidance[item["scheme_id"]] = DocumentGuidance(
                scheme_id=item["scheme_id"],
                required_documents=item["known_required_documents"],
                application_steps=[f"Visit or contact: {item['how_to_apply']}"] if item["how_to_apply"] else [],
                common_blockers=[],
            )

    state["document_guidance"] = guidance
    return state
