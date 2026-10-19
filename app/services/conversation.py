"""Conversation data service (delegates to unified data_service)."""

from services.data_service import (
    load_schema,
    load_conversations,
    load_existing_annotations,
    save_annotations,
    _merge_annotations as merge_annotations_to_conversation,
)


def save_conversation(conversation, filepath=None):
    """Save conversation to JSON file."""
    if filepath:
        conversation["filepath"] = str(filepath)
    return save_annotations(conversation, {}, expert_id=None)


__all__ = [
    "load_schema",
    "load_conversations",
    "save_conversation",
    "merge_annotations_to_conversation",
    "load_existing_annotations",
]
