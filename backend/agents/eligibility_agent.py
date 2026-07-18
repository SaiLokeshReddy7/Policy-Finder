"""Eligibility Reasoning Agent: scores each candidate scheme against the
citizen profile using Claude, with structured JSON output."""
from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.core.config import get_model_settings
from backend.llm.anthropic_client import complete_json
from backend.models.schemas import EligibilityResult, EligibilityVerdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Eligibility Reasoning Agent of a public welfare-scheme
navigator. Given a citizen's profile and a list of candidate government schemes, decide
for EACH scheme whether the citizen is:
- "Likely Eligible": profile clearly satisfies the scheme's stated eligibility rules
- "Possibly Eligible": partially matches, or some required fields are missing/ambiguous
- "Not Eligible": profile clearly violates a hard eligibility rule (age, gender, income cap, etc.)
- "Needs More Info": the scheme's rules can't be evaluated confidently from the given profile

For each scheme return a JSON object with keys: scheme_id, verdict, reason (1-2 plain
sentences explaining WHY, referencing the specific profile attribute that drove the
decision), confidence (0.0-1.0). Return ONLY a JSON array of these objects -- no prose,
no markdown code fences, no extra keys."""


def _fallback_result(scheme_id: str, scheme_name: str) -> EligibilityResult:
    return EligibilityResult(
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        verdict=EligibilityVerdict.needs_more_info,
        reason="Automated eligibility reasoning was unavailable; please verify manually against the official source.",
        confidence=0.0,
    )


def run_eligibility_agent(state: AgentState) -> AgentState:
    profile = state["profile"]
    candidates = state.get("candidates", [])
    settings = get_model_settings().anthropic

    if not candidates:
        state["eligibility_results"] = []
        return state

    scheme_payload = [
        {
            "scheme_id": c.scheme_id,
            "name": c.name,
            "eligibility_rules": c.eligibility_rules,
            "description": c.description,
        }
        for c in candidates
    ]
    user_prompt = (
        f"Citizen profile summary: {state.get('profile_summary', '')}\n"
        f"Structured profile (JSON): {profile.model_dump_json()}\n"
        f"Candidate schemes (JSON): {scheme_payload}"
    )

    results: list[EligibilityResult] = []
    try:
        raw_results = complete_json(
            model=settings.models.eligibility_reasoning,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
        )
        by_id = {c.scheme_id: c for c in candidates}
        for item in raw_results:
            scheme_id = item.get("scheme_id")
            candidate = by_id.get(scheme_id)
            if candidate is None:
                continue
            try:
                verdict = EligibilityVerdict(item.get("verdict", "Needs More Info"))
            except ValueError:
                verdict = EligibilityVerdict.needs_more_info
            results.append(
                EligibilityResult(
                    scheme_id=scheme_id,
                    scheme_name=candidate.name,
                    verdict=verdict,
                    reason=item.get("reason", ""),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        covered_ids = {r.scheme_id for r in results}
        for candidate in candidates:
            if candidate.scheme_id not in covered_ids:
                results.append(_fallback_result(candidate.scheme_id, candidate.name))
    except Exception:
        logger.warning("Eligibility agent LLM call failed; returning fallback verdicts", exc_info=True)
        results = [_fallback_result(c.scheme_id, c.name) for c in candidates]

    state["eligibility_results"] = results
    return state
