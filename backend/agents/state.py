"""Shared state threaded through the LangGraph navigator pipeline."""
from __future__ import annotations

from typing import Optional, TypedDict

from backend.models.schemas import (
    CitizenProfile,
    DocumentGuidance,
    EligibilityResult,
    SchemeCandidate,
    SchemeRecommendation,
)


class AgentState(TypedDict, total=False):
    profile: CitizenProfile
    free_text_context: Optional[str]
    language: str
    profile_summary: str
    candidates: list[SchemeCandidate]
    eligibility_results: list[EligibilityResult]
    document_guidance: dict[str, DocumentGuidance]
    simplified_explanations: dict[str, str]
    recommendations: list[SchemeRecommendation]
    warnings: list[str]
