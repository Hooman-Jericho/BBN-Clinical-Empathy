"""Pydantic schemas for structured LLM outputs, ReAct tools, and multi-agent consensus."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SpanPredictionSchema(BaseModel):
    """Pydantic model for a single predicted span."""
    text: str = Field(description="Exact span substring from the turn text")
    start: int = Field(default=0, ge=0, description="Character start offset")
    end: int = Field(default=0, ge=0, description="Character end offset")
    label: str = Field(description="Clinical appraisal or elicitation/response label")
    reasoning: Optional[str] = Field(default="", description="Clinical rationale for this annotation")
    confidence: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")


class TurnAnnotationSchema(BaseModel):
    """Pydantic model for complete turn annotation output."""
    spikes_stage: Optional[str] = Field(default=None, description="Primary SPIKES protocol stage (clinician only)")
    annotations: List[SpanPredictionSchema] = Field(default_factory=list, description="List of span annotations")
    clinical_summary: Optional[str] = Field(default="", description="High-level summary of empathic exchange")


class RelationPredictionSchema(BaseModel):
    """Pydantic model for a cross-turn semantic relation."""
    from_span_id: str = Field(description="Source span ID (typically clinician response/elicitation)")
    to_span_id: str = Field(description="Target span ID (typically patient empathic opportunity)")
    relation_type: Literal["response_to", "elicitation_of"] = Field(description="Type of semantic relation")
    reasoning: Optional[str] = Field(default="", description="Semantic justification for linking")


class RelationListSchema(BaseModel):
    """Pydantic model for multiple relation predictions."""
    relations: List[RelationPredictionSchema] = Field(default_factory=list)


class ReActToolCallSchema(BaseModel):
    """Pydantic model for tool call in ReAct execution loop."""
    thought: str = Field(description="Internal reasoning about current state and next step")
    action: Literal["schema_lookup", "text_search", "validate_span", "finish"] = Field(description="Tool to call")
    action_input: dict = Field(default_factory=dict, description="Arguments for tool call")


class CoordinatorPlanSchema(BaseModel):
    """Pydantic model for multi-agent coordinator dispatch plan."""
    speaker_role: str = Field(description="Detected speaker: patient or clinician")
    active_specialists: List[str] = Field(description="Specialist agents to dispatch to")
    coordination_strategy: str = Field(description="Guidance for clinical analysis")


class CriticReviewSchema(BaseModel):
    """Pydantic model for consensus critic review."""
    approved_spans: List[SpanPredictionSchema] = Field(default_factory=list, description="Validated spans")
    rejected_spans: List[SpanPredictionSchema] = Field(default_factory=list, description="Rejected false positives")
    critic_notes: Optional[str] = Field(default="", description="Critique notes on clinical consistency")
