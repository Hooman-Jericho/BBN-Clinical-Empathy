"""Database models (dataclasses) for BBN Annotation Tool."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class Expert:
    """Expert / Annotator model."""
    id: Optional[int] = None
    name: str = ""
    email: Optional[str] = None
    role: str = "annotator"  # annotator, reviewer, admin
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Optional["Expert"]:
        """Create Expert from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            role=row["role"],
            created_at=row["created_at"],
        )


@dataclass
class Conversation:
    """Conversation model."""
    id: Optional[int] = None
    external_id: str = ""
    scenario: Optional[str] = None
    language: str = "en"
    date: Optional[str] = None
    source_file: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: Optional[datetime] = None
    turns: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row) -> Optional["Conversation"]:
        """Create Conversation from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            external_id=row["external_id"],
            scenario=row["scenario"],
            language=row["language"],
            date=row["date"],
            source_file=row["source_file"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.external_id,
            "metadata": {
                "scenario": self.scenario,
                "language": self.language,
                "date": self.date,
            },
            "turns": [t.to_dict() for t in self.turns] if self.turns else [],
            "db_id": self.id,
            "source_file": self.source_file,
        }


@dataclass
class Turn:
    """Conversation turn model."""
    id: Optional[int] = None
    conversation_id: Optional[int] = None
    turn_number: int = 0
    speaker: str = ""
    text: str = ""
    annotations: dict = field(default_factory=dict)

    @classmethod
    def from_row(cls, row) -> Optional["Turn"]:
        """Create Turn from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            conversation_id=row["conversation_id"],
            turn_number=row["turn_number"],
            speaker=row["speaker"],
            text=row["text"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "turn_id": self.turn_number,
            "speaker": self.speaker,
            "text": self.text,
            "annotations": self.annotations,
            "db_id": self.id,
        }


@dataclass
class SpanAnnotation:
    """Span annotation model."""
    id: Optional[int] = None
    turn_id: Optional[int] = None
    expert_id: Optional[int] = None
    span_id: str = ""
    text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    label: str = ""
    source: str = "manual"  # manual, ai_accepted, ai_modified, imported
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Optional["SpanAnnotation"]:
        """Create SpanAnnotation from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            expert_id=row["expert_id"],
            span_id=row["span_id"],
            text=row["text"],
            start_pos=row["start_pos"],
            end_pos=row["end_pos"],
            label=row["label"],
            source=row["source"],
            confidence=row["confidence"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start_pos,
            "end": self.end_pos,
            "label": self.label,
            "source": self.source,
            "expert_id": self.expert_id,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data


@dataclass
class Relation:
    """Relation between spans model."""
    id: Optional[int] = None
    turn_id: Optional[int] = None
    expert_id: Optional[int] = None
    relation_id: str = ""
    from_span_id: str = ""
    to_span_id: str = ""
    to_turn_id: Optional[int] = None
    relation_type: str = ""
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Optional["Relation"]:
        """Create Relation from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            expert_id=row["expert_id"],
            relation_id=row["relation_id"],
            from_span_id=row["from_span_id"],
            to_span_id=row["to_span_id"],
            to_turn_id=row["to_turn_id"],
            relation_type=row["relation_type"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "relation_id": self.relation_id,
            "from": self.from_span_id,
            "to": self.to_span_id,
            "to_turn_id": self.to_turn_id,
            "type": self.relation_type,
        }


@dataclass
class SpikesAnnotation:
    """SPIKES stage annotation model."""
    id: Optional[int] = None
    turn_id: Optional[int] = None
    expert_id: Optional[int] = None
    stage: str = ""
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Optional["SpikesAnnotation"]:
        """Create SpikesAnnotation from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            expert_id=row["expert_id"],
            stage=row["stage"],
            created_at=row["created_at"],
        )


@dataclass
class AISuggestion:
    """AI suggestion model (for tracking agent performance and human reviews)."""
    id: Optional[int] = None
    turn_id: Optional[int] = None
    span_id: str = ""
    text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    suggested_label: str = ""
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    agent_type: Optional[str] = None
    model: Optional[str] = None
    status: str = "pending"  # pending, accepted, rejected, modified
    expert_id: Optional[int] = None
    created_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> Optional["AISuggestion"]:
        """Create AISuggestion from database row."""
        if row is None:
            return None
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            span_id=row["span_id"],
            text=row["text"],
            start_pos=row["start_pos"],
            end_pos=row["end_pos"],
            suggested_label=row["suggested_label"],
            confidence=row["confidence"],
            reasoning=row["reasoning"] if "reasoning" in row.keys() else None,
            agent_type=row["agent_type"],
            model=row["model"],
            status=row["status"],
            expert_id=row["expert_id"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start_pos,
            "end": self.end_pos,
            "suggested_label": self.suggested_label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "agent_type": self.agent_type,
            "model": self.model,
            "status": self.status,
        }
