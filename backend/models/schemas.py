"""Pydantic domain models shared by the API layer and the LangGraph agents."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class SocialCategory(str, Enum):
    general = "General"
    sc = "SC"
    st = "ST"
    obc = "OBC"
    minority = "Minority"


class EligibilityVerdict(str, Enum):
    likely_eligible = "Likely Eligible"
    possibly_eligible = "Possibly Eligible"
    not_eligible = "Not Eligible"
    needs_more_info = "Needs More Info"


class CitizenProfile(BaseModel):
    """Structured household profile collected from the Streamlit intake form."""

    age: int = Field(..., ge=0, le=120)
    gender: Gender
    annual_income: int = Field(..., ge=0, description="Household annual income in INR")
    occupation: str = ""
    state: str = ""
    category: SocialCategory = SocialCategory.general
    is_student: bool = False
    is_disabled: bool = False
    is_farmer: bool = False
    is_bpl: bool = False
    family_size: int = Field(default=1, ge=1)


class NavigateRequest(BaseModel):
    profile: CitizenProfile
    language: str = "en"
    free_text_context: Optional[str] = Field(
        default=None,
        description="Optional free-text description of the citizen's situation, in any language.",
    )


class SchemeSource(BaseModel):
    name: str
    url: str
    origin: str = Field(description="'knowledge_base' or 'web_search'")


class SchemeCandidate(BaseModel):
    scheme_id: str
    name: str
    category: str
    description: str
    benefits: str
    eligibility_rules: dict = Field(default_factory=dict)
    required_documents: list[str] = Field(default_factory=list)
    how_to_apply: str = ""
    source: SchemeSource


class EligibilityResult(BaseModel):
    scheme_id: str
    scheme_name: str
    verdict: EligibilityVerdict
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DocumentGuidance(BaseModel):
    scheme_id: str
    required_documents: list[str] = Field(default_factory=list)
    application_steps: list[str] = Field(default_factory=list)
    common_blockers: list[str] = Field(default_factory=list)


class SchemeRecommendation(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    verdict: EligibilityVerdict
    reason: str
    confidence: float
    benefits: str
    required_documents: list[str] = Field(default_factory=list)
    application_steps: list[str] = Field(default_factory=list)
    common_blockers: list[str] = Field(default_factory=list)
    simplified_explanation: str = ""
    source: SchemeSource


class NavigateResponse(BaseModel):
    profile_summary: str
    language: str
    recommendations: list[SchemeRecommendation]
    generated_at: str
    warnings: list[str] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """Follow-up question about a set of recommendations the user already received."""

    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    profile_summary: str = ""
    recommendations: list[SchemeRecommendation] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


class SchemeSummary(BaseModel):
    """Lightweight scheme listing used by GET /api/v1/schemes."""

    scheme_id: str
    name: str
    category: str
    description: str
    source: SchemeSource


class HealthStatus(BaseModel):
    status: str
    vector_store_loaded: bool
    schemes_indexed: int
    anthropic_configured: bool
    huggingface_configured: bool
    search_provider: str


class VoiceTranscribeResponse(BaseModel):
    transcript: str
    suggested_profile: dict = Field(
        default_factory=dict,
        description="Best-effort CitizenProfile fields extracted from the transcript; only confidently-inferred keys are present.",
    )


class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = "en"
