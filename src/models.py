"""Data models for KASPARRO FSE Intern Pathway Assessment.

Defines schemas for catalogue facts, answer records, gold claims,
predicted claims, evaluation metrics, and API request/response models.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


VALID_LABELS = {"supported", "contradicted", "unknown"}
VALID_BRANDS = {"cleartrail", "stonewave", "cloudnote"}
VALID_SPLITS = {"development", "check"}
VALID_VERSIONS = {"v1", "v2"}


class BrandFact(BaseModel):
    """Fact row from brand_facts.csv."""
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: Any
    value_type: str
    unit: str = ""
    condition: str = ""
    valid_from: str
    valid_to: str


class AnswerRecord(BaseModel):
    """Answer row from answers.csv."""
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


class GoldClaim(BaseModel):
    """Gold reference claim row from development_gold.csv."""
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: List[str] = Field(default_factory=list)
    explanation: str = ""
    difficulty_tags: List[str] = Field(default_factory=list)

    @field_validator("expected_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Invalid gold label: {v}")
        return v


class RawExtractedClaim(BaseModel):
    """Raw claim returned by model before character span derivation."""
    quote: str
    label: str
    fact_ids: List[str] = Field(default_factory=list)
    reason: str = ""


class PredictedClaim(BaseModel):
    """Validated predicted claim with exact span offsets."""
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: List[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Invalid predicted claim label: {v}")
        return v


class AnswerResult(BaseModel):
    """Terminal outcome for an individual answer."""
    answer_id: str
    status: str  # 'success' | 'error'
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    claims: List[PredictedClaim] = Field(default_factory=list)
    turns_used: int = 0
    trace_ref: Optional[str] = None
    model_exchange: Optional[List[dict]] = None


class ClassifyRequest(BaseModel):
    """Request schema for POST /classify."""
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: str = "v1"

    @field_validator("agent_version")
    @classmethod
    def validate_agent_version(cls, v: str) -> str:
        if v not in VALID_VERSIONS:
            raise ValueError(f"agent_version must be one of {VALID_VERSIONS}")
        return v

    @field_validator("brand_id")
    @classmethod
    def validate_brand_id(cls, v: str) -> str:
        if v not in VALID_BRANDS:
            raise ValueError(f"Unknown brand_id: {v}")
        return v


class ClassifyResponse(BaseModel):
    """Response schema for POST /classify."""
    answer_id: str
    agent_version: str
    claims: List[PredictedClaim]
    latency_ms: float


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics."""
    G: Optional[int] = None
    P: Optional[int] = None
    M: Optional[int] = None
    C: Optional[int] = None
    extraction_precision: Optional[float] = None
    extraction_recall: Optional[float] = None
    extraction_f1: Optional[float] = None
    joint_precision: Optional[float] = None
    joint_recall: Optional[float] = None
    joint_f1: Optional[float] = None
    coverage: Optional[float] = None
    missed_gold_ids: Optional[List[str]] = None
    unmatched_prediction_ids: Optional[List[str]] = None
    matched_wrong_label_ids: Optional[List[dict]] = None
    error_count: int = 0


class EvaluationRunRequest(BaseModel):
    """Request schema for POST /evaluation-runs."""
    agent_version: str = "v1"
    split: str = "development"

    @field_validator("agent_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v not in VALID_VERSIONS:
            raise ValueError(f"agent_version must be one of {VALID_VERSIONS}")
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}")
        return v


class EvaluationRunResponse(BaseModel):
    """Response schema for evaluation run endpoints."""
    run_id: str
    status: str
    agent_version: str
    split: str
    answer_count: int
    accepted_prediction_count: int
    error_count: int
    metrics: Optional[EvaluationMetrics] = None
    created_at: Optional[str] = None
    duration_seconds: Optional[float] = None
