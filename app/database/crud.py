"""CRUD operations for BBN Annotation Tool database with full state synchronization."""

import json
import uuid
from typing import Optional, List, Dict, Any
from database.connection import get_db
from database.models import (
    Expert,
    Conversation,
    Turn,
    SpanAnnotation,
    Relation,
    SpikesAnnotation,
    AISuggestion,
)


# ============== EXPERTS ==============

def create_expert(name: str, email: str = None, role: str = "annotator") -> Expert:
    """Create a new expert."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO experts (name, email, role) VALUES (?, ?, ?)",
            (name, email, role)
        )
        expert_id = cursor.lastrowid
        return Expert(id=expert_id, name=name, email=email, role=role)


def get_expert(expert_id: int) -> Optional[Expert]:
    """Get expert by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experts WHERE id = ?", (expert_id,))
        row = cursor.fetchone()
        return Expert.from_row(row)


def get_expert_by_name(name: str) -> Optional[Expert]:
    """Get expert by name."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experts WHERE name = ?", (name,))
        row = cursor.fetchone()
        return Expert.from_row(row)


def get_all_experts() -> List[Expert]:
    """Get all experts."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experts ORDER BY name")
        return [Expert.from_row(row) for row in cursor.fetchall()]


def get_or_create_expert(name: str, email: str = None) -> Expert:
    """Get existing expert or create new one."""
    expert = get_expert_by_name(name)
    if expert:
        return expert
    return create_expert(name, email)


# ============== CONVERSATIONS ==============

def create_conversation(
    external_id: str,
    scenario: str = None,
    language: str = "en",
    date: str = None,
    source_file: str = None,
    metadata: dict = None,
) -> Conversation:
    """Create a new conversation."""
    with get_db() as conn:
        cursor = conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        cursor.execute(
            """INSERT INTO conversations
               (external_id, scenario, language, date, source_file, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (external_id, scenario, language, date, source_file, metadata_json)
        )
        conv_id = cursor.lastrowid
        return Conversation(
            id=conv_id,
            external_id=external_id,
            scenario=scenario,
            language=language,
            date=date,
            source_file=source_file,
            metadata_json=metadata_json,
        )


def get_conversation(conv_id: int) -> Optional[Conversation]:
    """Get conversation by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        return Conversation.from_row(row)


def get_conversation_by_external_id(external_id: str) -> Optional[Conversation]:
    """Get conversation by external ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE external_id = ?", (external_id,))
        row = cursor.fetchone()
        return Conversation.from_row(row)


def get_all_conversations() -> List[Conversation]:
    """Get all conversations."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        return [Conversation.from_row(row) for row in cursor.fetchall()]


# ============== TURNS ==============

def create_turn(
    conversation_id: int,
    turn_number: int,
    speaker: str,
    text: str,
) -> Turn:
    """Create a new turn."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO turns (conversation_id, turn_number, speaker, text)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, turn_number, speaker, text)
        )
        turn_id = cursor.lastrowid
        return Turn(
            id=turn_id,
            conversation_id=conversation_id,
            turn_number=turn_number,
            speaker=speaker,
            text=text,
        )


def get_turn(turn_id: int) -> Optional[Turn]:
    """Get turn by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM turns WHERE id = ?", (turn_id,))
        row = cursor.fetchone()
        return Turn.from_row(row)


def get_turns_by_conversation(conversation_id: int) -> List[Turn]:
    """Get all turns for a conversation in order."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_number",
            (conversation_id,)
        )
        return [Turn.from_row(row) for row in cursor.fetchall()]


# ============== SPAN ANNOTATIONS ==============

def create_span_annotation(
    turn_id: int,
    text: str,
    start_pos: int,
    end_pos: int,
    label: str,
    expert_id: int = None,
    source: str = "manual",
    confidence: float = None,
    span_id: str = None,
) -> SpanAnnotation:
    """Create a new span annotation."""
    if span_id is None:
        span_id = f"span_{uuid.uuid4().hex[:8]}"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO span_annotations
               (turn_id, expert_id, span_id, text, start_pos, end_pos, label, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (turn_id, expert_id, span_id, text, start_pos, end_pos, label, source, confidence)
        )
        annotation_id = cursor.lastrowid
        return SpanAnnotation(
            id=annotation_id,
            turn_id=turn_id,
            expert_id=expert_id,
            span_id=span_id,
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            label=label,
            source=source,
            confidence=confidence,
        )


def get_span_annotations(turn_id: int, expert_id: int = None) -> List[SpanAnnotation]:
    """Get span annotations for a turn."""
    with get_db() as conn:
        cursor = conn.cursor()
        if expert_id:
            cursor.execute(
                "SELECT * FROM span_annotations WHERE turn_id = ? AND expert_id = ?",
                (turn_id, expert_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM span_annotations WHERE turn_id = ?",
                (turn_id,)
            )
        return [SpanAnnotation.from_row(row) for row in cursor.fetchall()]


def delete_span_annotation(span_id: str) -> bool:
    """Delete a span annotation and cascade delete related relations."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM span_annotations WHERE span_id = ?", (span_id,))
        spans_deleted = cursor.rowcount > 0
        cursor.execute(
            "DELETE FROM relations WHERE from_span_id = ? OR to_span_id = ?",
            (span_id, span_id)
        )
        return spans_deleted


# ============== RELATIONS ==============

def create_relation(
    turn_id: int,
    from_span_id: str,
    to_span_id: str,
    relation_type: str,
    to_turn_id: int = None,
    expert_id: int = None,
    relation_id: str = None,
) -> Relation:
    """Create a new relation."""
    if relation_id is None:
        relation_id = f"rel_{uuid.uuid4().hex[:8]}"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO relations
               (turn_id, expert_id, relation_id, from_span_id, to_span_id, to_turn_id, relation_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (turn_id, expert_id, relation_id, from_span_id, to_span_id, to_turn_id, relation_type)
        )
        rel_id = cursor.lastrowid
        return Relation(
            id=rel_id,
            turn_id=turn_id,
            expert_id=expert_id,
            relation_id=relation_id,
            from_span_id=from_span_id,
            to_span_id=to_span_id,
            to_turn_id=to_turn_id,
            relation_type=relation_type,
        )


def get_relations(turn_id: int, expert_id: int = None) -> List[Relation]:
    """Get relations for a turn."""
    with get_db() as conn:
        cursor = conn.cursor()
        if expert_id:
            cursor.execute(
                "SELECT * FROM relations WHERE turn_id = ? AND expert_id = ?",
                (turn_id, expert_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM relations WHERE turn_id = ?",
                (turn_id,)
            )
        return [Relation.from_row(row) for row in cursor.fetchall()]


def delete_relation(relation_id: str) -> bool:
    """Delete a relation by relation_id."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relations WHERE relation_id = ?", (relation_id,))
        return cursor.rowcount > 0


# ============== SPIKES ANNOTATIONS ==============

def set_spikes_annotation(
    turn_id: int,
    stage: Optional[str],
    expert_id: int = None,
) -> Optional[SpikesAnnotation]:
    """Set or remove SPIKES stage for a turn (upsert / delete)."""
    with get_db() as conn:
        cursor = conn.cursor()
        if not stage or stage.lower() in ("none", ""):
            # Clear stage
            if expert_id:
                cursor.execute(
                    "DELETE FROM spikes_annotations WHERE turn_id = ? AND expert_id = ?",
                    (turn_id, expert_id)
                )
            else:
                cursor.execute(
                    "DELETE FROM spikes_annotations WHERE turn_id = ?",
                    (turn_id,)
                )
            return None
        else:
            cursor.execute(
                """INSERT INTO spikes_annotations (turn_id, expert_id, stage)
                   VALUES (?, ?, ?)
                   ON CONFLICT(turn_id, expert_id) DO UPDATE SET stage = excluded.stage""",
                (turn_id, expert_id, stage)
            )
            return SpikesAnnotation(turn_id=turn_id, expert_id=expert_id, stage=stage)


def get_spikes_annotation(turn_id: int, expert_id: int = None) -> Optional[SpikesAnnotation]:
    """Get SPIKES annotation for a turn."""
    with get_db() as conn:
        cursor = conn.cursor()
        if expert_id:
            cursor.execute(
                "SELECT * FROM spikes_annotations WHERE turn_id = ? AND expert_id = ?",
                (turn_id, expert_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM spikes_annotations WHERE turn_id = ? ORDER BY created_at DESC LIMIT 1",
                (turn_id,)
            )
        row = cursor.fetchone()
        return SpikesAnnotation.from_row(row)


# ============== AI SUGGESTIONS TRACKING ==============

def create_ai_suggestion(
    turn_id: int,
    span_id: str,
    text: str,
    start_pos: int,
    end_pos: int,
    suggested_label: str,
    confidence: Optional[float] = None,
    reasoning: Optional[str] = None,
    agent_type: Optional[str] = None,
    model: Optional[str] = None,
) -> AISuggestion:
    """Save an AI suggestion to the database for performance tracking."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO ai_suggestions
               (turn_id, span_id, text, start_pos, end_pos, suggested_label, confidence, reasoning, agent_type, model, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (turn_id, span_id, text, start_pos, end_pos, suggested_label, confidence, reasoning, agent_type, model)
        )
        suggestion_id = cursor.lastrowid
        return AISuggestion(
            id=suggestion_id,
            turn_id=turn_id,
            span_id=span_id,
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            suggested_label=suggested_label,
            confidence=confidence,
            reasoning=reasoning,
            agent_type=agent_type,
            model=model,
            status="pending",
        )


def update_ai_suggestion_status(
    turn_id: int,
    span_id: str,
    status: str,  # 'accepted', 'rejected', 'modified'
    expert_id: Optional[int] = None,
) -> bool:
    """Update human review status for an AI suggestion."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE ai_suggestions
               SET status = ?, expert_id = ?, reviewed_at = CURRENT_TIMESTAMP
               WHERE turn_id = ? AND span_id = ?""",
            (status, expert_id, turn_id, span_id)
        )
        return cursor.rowcount > 0


def get_ai_suggestions_for_turn(turn_id: int, status: Optional[str] = None) -> List[AISuggestion]:
    """Get all AI suggestions for a given turn."""
    with get_db() as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM ai_suggestions WHERE turn_id = ? AND status = ? ORDER BY id DESC",
                (turn_id, status)
            )
        else:
            cursor.execute(
                "SELECT * FROM ai_suggestions WHERE turn_id = ? ORDER BY id DESC",
                (turn_id,)
            )
        return [AISuggestion.from_row(row) for row in cursor.fetchall()]


# ============== OPTIMIZED BULK / BATCH OPERATIONS ==============

def get_full_conversation_with_annotations(
    conversation_id: int,
    expert_id: int = None,
) -> Optional[Conversation]:
    """Get conversation with all turns and annotations in a single optimized batch transaction.
    
    Eliminates N+1 query overhead by fetching all turns, spans, relations, and SPIKES stages
    using IN-clause batch queries across a single SQLite connection.
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # 1. Fetch conversation
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        conv_row = cursor.fetchone()
        if not conv_row:
            return None
        conv = Conversation.from_row(conv_row)

        # 2. Fetch turns
        cursor.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_number",
            (conversation_id,)
        )
        turns = [Turn.from_row(row) for row in cursor.fetchall()]
        if not turns:
            conv.turns = []
            return conv

        turn_ids = [t.id for t in turns]
        turn_id_to_turn = {t.id: t for t in turns}
        placeholders = ",".join("?" for _ in turn_ids)

        # Initialize annotations
        for turn in turns:
            turn.annotations = {"spans": [], "relations": [], "spikes_stage": None}

        # 3. Batch fetch spans
        if expert_id:
            cursor.execute(
                f"SELECT * FROM span_annotations WHERE turn_id IN ({placeholders}) AND expert_id = ?",
                turn_ids + [expert_id]
            )
        else:
            cursor.execute(
                f"SELECT * FROM span_annotations WHERE turn_id IN ({placeholders})",
                turn_ids
            )
        for row in cursor.fetchall():
            span = SpanAnnotation.from_row(row)
            if span.turn_id in turn_id_to_turn:
                turn_id_to_turn[span.turn_id].annotations["spans"].append(span.to_dict())

        # 4. Batch fetch relations
        if expert_id:
            cursor.execute(
                f"SELECT * FROM relations WHERE turn_id IN ({placeholders}) AND expert_id = ?",
                turn_ids + [expert_id]
            )
        else:
            cursor.execute(
                f"SELECT * FROM relations WHERE turn_id IN ({placeholders})",
                turn_ids
            )
        for row in cursor.fetchall():
            rel = Relation.from_row(row)
            if rel.turn_id in turn_id_to_turn:
                turn_id_to_turn[rel.turn_id].annotations["relations"].append(rel.to_dict())

        # 5. Batch fetch SPIKES
        if expert_id:
            cursor.execute(
                f"SELECT * FROM spikes_annotations WHERE turn_id IN ({placeholders}) AND expert_id = ?",
                turn_ids + [expert_id]
            )
        else:
            cursor.execute(
                f"SELECT * FROM spikes_annotations WHERE turn_id IN ({placeholders})",
                turn_ids
            )
        for row in cursor.fetchall():
            spikes = SpikesAnnotation.from_row(row)
            if spikes.turn_id in turn_id_to_turn:
                turn_id_to_turn[spikes.turn_id].annotations["spikes_stage"] = spikes.stage

        conv.turns = turns
        return conv


def sync_session_annotations(
    conversation_id: int,
    annotations: Dict[int, Dict[str, Any]],
    expert_id: int = None,
) -> bool:
    """Synchronize session annotations with the database (Diff & Sync).
    
    Ensures absolute data integrity:
    - Deletes spans and relations that were removed in the session.
    - Updates spans whose positions or labels were modified.
    - Inserts newly created spans and relations.
    - Updates or removes SPIKES stages.
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Get all turns for this conversation
        cursor.execute(
            "SELECT id, turn_number FROM turns WHERE conversation_id = ?",
            (conversation_id,)
        )
        turns_map = {row["turn_number"]: row["id"] for row in cursor.fetchall()}

        for turn_number, turn_id in turns_map.items():
            session_data = annotations.get(turn_number, {"spans": [], "relations": [], "spikes_stage": None})

            # --- 1. SPANS SYNC ---
            cursor.execute("SELECT * FROM span_annotations WHERE turn_id = ?", (turn_id,))
            db_spans = {row["span_id"]: row for row in cursor.fetchall()}
            session_spans = session_data.get("spans", [])
            session_span_ids = {s["span_id"] for s in session_spans if "span_id" in s}

            # Delete spans that were removed in session
            for span_id in list(db_spans.keys()):
                if span_id not in session_span_ids:
                    cursor.execute("DELETE FROM span_annotations WHERE span_id = ?", (span_id,))
                    cursor.execute("DELETE FROM relations WHERE from_span_id = ? OR to_span_id = ?", (span_id, span_id))

            # Insert or update session spans
            for s in session_spans:
                s_id = s.get("span_id", f"span_{uuid.uuid4().hex[:8]}")
                s["span_id"] = s_id
                if s_id in db_spans:
                    cursor.execute(
                        """UPDATE span_annotations
                           SET text = ?, start_pos = ?, end_pos = ?, label = ?, source = ?, confidence = ?
                           WHERE span_id = ?""",
                        (s["text"], s["start"], s["end"], s["label"], s.get("source", "manual"), s.get("confidence"), s_id)
                    )
                else:
                    cursor.execute(
                        """INSERT INTO span_annotations
                           (turn_id, expert_id, span_id, text, start_pos, end_pos, label, source, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (turn_id, expert_id, s_id, s["text"], s["start"], s["end"], s["label"], s.get("source", "manual"), s.get("confidence"))
                    )

            # --- 2. RELATIONS SYNC ---
            cursor.execute("SELECT * FROM relations WHERE turn_id = ?", (turn_id,))
            db_relations = {row["relation_id"]: row for row in cursor.fetchall()}
            session_relations = session_data.get("relations", [])
            session_rel_ids = {r.get("relation_id") for r in session_relations if r.get("relation_id")}

            # Delete relations removed in session
            for rel_id in list(db_relations.keys()):
                if rel_id not in session_rel_ids:
                    cursor.execute("DELETE FROM relations WHERE relation_id = ?", (rel_id,))

            # Insert new relations
            for r in session_relations:
                rel_id = r.get("relation_id", f"rel_{uuid.uuid4().hex[:8]}")
                r["relation_id"] = rel_id
                if rel_id not in db_relations:
                    cursor.execute(
                        """INSERT INTO relations
                           (turn_id, expert_id, relation_id, from_span_id, to_span_id, to_turn_id, relation_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (turn_id, expert_id, rel_id, r["from"], r["to"], r.get("to_turn_id"), r["type"])
                    )

            # --- 3. SPIKES STAGE SYNC ---
            stage = session_data.get("spikes_stage")
            if stage and stage.lower() not in ("none", ""):
                cursor.execute(
                    """INSERT INTO spikes_annotations (turn_id, expert_id, stage)
                       VALUES (?, ?, ?)
                       ON CONFLICT(turn_id, expert_id) DO UPDATE SET stage = excluded.stage""",
                    (turn_id, expert_id, stage)
                )
            else:
                cursor.execute(
                    "DELETE FROM spikes_annotations WHERE turn_id = ?",
                    (turn_id,)
                )

        return True


# Backward-compatible alias
save_session_annotations = sync_session_annotations


def import_conversation_from_json(json_data: dict, source_file: str = None) -> Conversation:
    """Import a conversation and its annotations from JSON format into SQLite."""
    external_id = json_data.get("id", f"conv_{uuid.uuid4().hex[:8]}")
    metadata = json_data.get("metadata", {})

    existing = get_conversation_by_external_id(external_id)
    if existing:
        return existing

    conv = create_conversation(
        external_id=external_id,
        scenario=metadata.get("scenario"),
        language=metadata.get("language", "en"),
        date=metadata.get("date"),
        source_file=source_file,
        metadata=metadata,
    )

    for turn_data in json_data.get("turns", []):
        turn = create_turn(
            conversation_id=conv.id,
            turn_number=turn_data["turn_id"],
            speaker=turn_data["speaker"],
            text=turn_data["text"],
        )

        annotations = turn_data.get("annotations", {})

        for span in annotations.get("spans", []):
            create_span_annotation(
                turn_id=turn.id,
                text=span["text"],
                start_pos=span.get("start", 0),
                end_pos=span.get("end", len(span["text"])),
                label=span["label"],
                span_id=span.get("span_id"),
                source="imported",
            )

        for rel in annotations.get("relations", []):
            create_relation(
                turn_id=turn.id,
                from_span_id=rel["from"],
                to_span_id=rel["to"],
                relation_type=rel["type"],
                to_turn_id=rel.get("to_turn_id"),
                relation_id=rel.get("relation_id"),
            )

        if annotations.get("spikes_stage"):
            set_spikes_annotation(turn.id, annotations["spikes_stage"])

    return conv
