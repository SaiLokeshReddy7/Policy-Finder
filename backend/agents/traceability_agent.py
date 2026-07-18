"""Traceability Agent: deterministic pass that assembles the final ranked
recommendations, each one carrying its originating source (KB record or web
search result URL) so every claim can be traced back to it."""
from __future__ import annotations

from backend.agents.state import AgentState
from backend.models.schemas import EligibilityVerdict, SchemeRecommendation

VERDICT_PRIORITY = {
    EligibilityVerdict.likely_eligible: 0,
    EligibilityVerdict.possibly_eligible: 1,
    EligibilityVerdict.needs_more_info: 2,
    EligibilityVerdict.not_eligible: 3,
}


def run_traceability_agent(state: AgentState) -> AgentState:
    candidates = {c.scheme_id: c for c in state.get("candidates", [])}
    guidance = state.get("document_guidance", {})
    explanations = state.get("simplified_explanations", {})
    eligibility_results = state.get("eligibility_results", [])

    recommendations: list[SchemeRecommendation] = []
    for r in eligibility_results:
        candidate = candidates.get(r.scheme_id)
        if candidate is None:
            continue
        doc = guidance.get(r.scheme_id)
        recommendations.append(
            SchemeRecommendation(
                scheme_id=r.scheme_id,
                scheme_name=candidate.name,
                category=candidate.category,
                verdict=r.verdict,
                reason=r.reason,
                confidence=r.confidence,
                benefits=candidate.benefits,
                required_documents=doc.required_documents if doc else candidate.required_documents,
                application_steps=doc.application_steps if doc else [],
                common_blockers=doc.common_blockers if doc else [],
                simplified_explanation=explanations.get(r.scheme_id, r.reason),
                source=candidate.source,
            )
        )

    recommendations.sort(key=lambda rec: (VERDICT_PRIORITY.get(rec.verdict, 9), -rec.confidence))
    state["recommendations"] = recommendations
    return state
