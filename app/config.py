"""Configuration constants for BBN Annotation Tool."""

from pathlib import Path

# Paths
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
SCHEMA_PATH = DATA_DIR / "annotation_schema.json"
STYLES_PATH = APP_DIR / "styles.css"

# Color mappings for labels
LABEL_COLORS = {
    # Patient EOs - Warm colors for feelings, cool for others
    "explicit_feeling": "#FF6B6B",
    "implicit_feeling": "#FFA07A",
    "explicit_appreciation": "#20E3B2",
    "implicit_appreciation": "#7DFFCC",
    "explicit_judgement": "#74B9FF",
    "implicit_judgement": "#A8D8FF",
    # Clinician Elicitations
    "direct_elicitation_feeling": "#FFD93D",
    "indirect_elicitation_feeling": "#FFE66D",
    "direct_elicitation_appreciation": "#FFA94D",
    "indirect_elicitation_appreciation": "#FFBE76",
    "direct_elicitation_judgement": "#FF9FF3",
    "indirect_elicitation_judgement": "#FFB8E0",
    # Clinician Responses - Acceptance
    "acceptance_positive_regard_explicit_judgement": "#00D9A5",
    "acceptance_positive_regard_implicit_judgement": "#55EFC4",
    "acceptance_positive_regard_repetition": "#81ECEC",
    "acceptance_positive_regard_allowing": "#B8F5F1",
    "acceptance_neutral_support_appreciation": "#FDCB6E",
    "acceptance_neutral_support_judgement": "#FFE5A0",
    # Clinician Responses - Sharing
    "sharing_feeling": "#E056FD",
    "sharing_appreciation": "#F0A1FF",
    "sharing_judgement": "#FFD1FF",
    # Clinician Responses - Understanding
    "understanding_feeling": "#54A0FF",
    "understanding_appreciation": "#82CCFF",
    "understanding_judgement": "#B8E0FF",
    # SPIKES Stages
    "setting": "#FF6B6B",
    "perception": "#FFA502",
    "invitation": "#FFD93D",
    "knowledge": "#2ED573",
    "empathy": "#54A0FF",
    "strategy": "#A55EEA",
}

DEFAULT_COLOR = "#95A5A6"

# Label categories organized by speaker type
PATIENT_LABELS = {
    "Feelings": ["explicit_feeling", "implicit_feeling"],
    "Appreciation": ["explicit_appreciation", "implicit_appreciation"],
    "Judgement": ["explicit_judgement", "implicit_judgement"],
}

CLINICIAN_LABELS = {
    "Elicitations": [
        "direct_elicitation_feeling",
        "indirect_elicitation_feeling",
        "direct_elicitation_appreciation",
        "indirect_elicitation_appreciation",
        "direct_elicitation_judgement",
        "indirect_elicitation_judgement",
    ],
    "Acceptance": [
        "acceptance_positive_regard_explicit_judgement",
        "acceptance_positive_regard_implicit_judgement",
        "acceptance_positive_regard_repetition",
        "acceptance_positive_regard_allowing",
        "acceptance_neutral_support_appreciation",
        "acceptance_neutral_support_judgement",
    ],
    "Sharing": ["sharing_feeling", "sharing_appreciation", "sharing_judgement"],
    "Understanding": [
        "understanding_feeling",
        "understanding_appreciation",
        "understanding_judgement",
    ],
}

SPIKES_STAGES = ["setting", "perception", "invitation", "knowledge", "empathy", "strategy"]

# Undo history settings
MAX_UNDO_HISTORY = 20

# Agent defaults
DEFAULT_AGENT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o",
    "agent_type": "react",
}

# OpenAI models
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

# Anthropic models
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]

# Database settings
USE_DATABASE = True  # Set to False to use JSON files only
DB_PATH = DATA_DIR / "annotations.db"
