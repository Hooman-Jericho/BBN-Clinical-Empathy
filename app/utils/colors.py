"""Color and label utility functions."""

from config import LABEL_COLORS, DEFAULT_COLOR, PATIENT_LABELS, CLINICIAN_LABELS


def get_label_color(label):
    """Get color for a label from config."""
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


def format_label_name(label):
    """Format label name for display."""
    return label.replace("_", " ").title()


def get_labels_for_speaker(speaker):
    """Get available labels for a speaker type."""
    if speaker == "patient":
        return PATIENT_LABELS
    return CLINICIAN_LABELS


def get_all_labels_flat(speaker):
    """Get all labels for a speaker as a flat list."""
    label_groups = get_labels_for_speaker(speaker)
    all_labels = []
    for labels in label_groups.values():
        all_labels.extend(labels)
    return all_labels
