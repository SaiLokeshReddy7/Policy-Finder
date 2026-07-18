from __future__ import annotations

from backend.agents.document_agent import run_document_agent
from backend.agents.eligibility_agent import run_eligibility_agent
from backend.agents.simplification_agent import run_simplification_agent
from backend.agents.traceability_agent import run_traceability_agent
from backend.models.schemas import CitizenProfile, SchemeCandidate, SchemeSource


def _make_profile() -> CitizenProfile:
    return CitizenProfile(
        age=45,
        gender="male",
        annual_income=120000,
        occupation="farmer",
        state="Bihar",
        category="General",
        is_farmer=True,
    )


def _make_candidate(sample_scheme: dict) -> SchemeCandidate:
    return SchemeCandidate(
        scheme_id=sample_scheme["id"],
        name=sample_scheme["name"],
        category=sample_scheme["category"],
        description=sample_scheme["description"],
        benefits=sample_scheme["benefits"],
        eligibility_rules=sample_scheme["eligibility"],
        required_documents=sample_scheme["required_documents"],
        how_to_apply=sample_scheme["how_to_apply"],
        source=SchemeSource(
            name=sample_scheme["source_name"], url=sample_scheme["source_url"], origin="knowledge_base"
        ),
    )


def test_full_agent_pipeline_produces_traceable_recommendation(mock_llm_calls, sample_scheme):
    candidate = _make_candidate(sample_scheme)
    state = {
        "profile": _make_profile(),
        "profile_summary": "45-year-old male farmer in Bihar with low income.",
        "language": "en",
        "candidates": [candidate],
        "warnings": [],
    }

    state = run_eligibility_agent(state)
    assert len(state["eligibility_results"]) == 1
    assert state["eligibility_results"][0].verdict.value == "Likely Eligible"

    state = run_document_agent(state)
    assert sample_scheme["id"] in state["document_guidance"]
    assert "Aadhaar card" in state["document_guidance"][sample_scheme["id"]].required_documents

    state = run_simplification_agent(state)
    assert state["simplified_explanations"][sample_scheme["id"]] == "Simplified test explanation."

    state = run_traceability_agent(state)
    recommendations = state["recommendations"]
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.scheme_id == sample_scheme["id"]
    assert rec.source.url == sample_scheme["source_url"]
    assert rec.source.origin == "knowledge_base"
    assert rec.simplified_explanation == "Simplified test explanation."


def test_eligibility_agent_falls_back_gracefully_on_llm_failure(monkeypatch, sample_scheme):
    import backend.agents.eligibility_agent as eligibility_agent

    def _boom(*args, **kwargs):
        raise RuntimeError("Anthropic API unreachable")

    monkeypatch.setattr(eligibility_agent, "complete_json", _boom)

    candidate = _make_candidate(sample_scheme)
    state = {
        "profile": _make_profile(),
        "profile_summary": "test",
        "candidates": [candidate],
    }
    state = run_eligibility_agent(state)
    results = state["eligibility_results"]
    assert len(results) == 1
    assert results[0].verdict.value == "Needs More Info"
    assert results[0].confidence == 0.0


def test_traceability_agent_sorts_by_verdict_priority():
    from backend.models.schemas import EligibilityResult, EligibilityVerdict

    candidates = [
        SchemeCandidate(
            scheme_id="a",
            name="Scheme A",
            category="Cat",
            description="",
            benefits="",
            source=SchemeSource(name="A", url="https://a.gov.in", origin="knowledge_base"),
        ),
        SchemeCandidate(
            scheme_id="b",
            name="Scheme B",
            category="Cat",
            description="",
            benefits="",
            source=SchemeSource(name="B", url="https://b.gov.in", origin="knowledge_base"),
        ),
    ]
    state = {
        "candidates": candidates,
        "document_guidance": {},
        "simplified_explanations": {},
        "eligibility_results": [
            EligibilityResult(scheme_id="a", scheme_name="Scheme A", verdict=EligibilityVerdict.not_eligible, reason="no", confidence=0.9),
            EligibilityResult(scheme_id="b", scheme_name="Scheme B", verdict=EligibilityVerdict.likely_eligible, reason="yes", confidence=0.6),
        ],
    }
    result_state = run_traceability_agent(state)
    ordered_ids = [r.scheme_id for r in result_state["recommendations"]]
    assert ordered_ids == ["b", "a"]
