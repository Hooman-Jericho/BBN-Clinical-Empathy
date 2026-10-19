# BBN Annotation Agent Documentation

This document describes the AI agent system integrated into the BBN Annotation Tool for automated annotation of Breaking Bad News (BBN) clinical conversations.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Agent Types](#agent-types)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Training Workflow](#training-workflow)

---

## Overview

The BBN Annotation Agent system provides AI-powered annotation suggestions for clinical conversations. It supports two agent architectures:

1. **ReAct Agent** - Single agent with step-by-step reasoning
2. **Multi-Agent System** - Specialized agents for different annotation tasks

### Key Features

- Automatic detection of Empathic Opportunities (EOs) in patient speech
- Classification of clinician responses and elicitations
- SPIKES protocol stage identification
- Relation linking between patient EOs and clinician responses
- Side-by-side comparison with expert annotations
- Support for OpenAI and Anthropic LLMs

---

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        ANNOTATION FLOW                           │
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│   │ Agent    │ ──► │ Expert   │ ──► │ Train &  │ ──► (repeat)   │
│   │ Proposes │     │ Reviews  │     │ Improve  │                │
│   └──────────┘     └──────────┘     └──────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
agents/
├── __init__.py          # Package exports
├── base.py              # Base classes, data types, label definitions
├── prompts.py           # System and user prompts for all agents
├── react_agent.py       # ReAct single-agent implementation
├── multi_agent.py       # Multi-agent system with specialists
└── runner.py            # Unified runner interface + CLI
```

---

## Agent Types

### 1. ReAct Agent

A single agent that uses the ReAct (Reasoning + Acting) pattern to annotate conversations step by step.

**Process:**
```
THOUGHT → ACTION → OBSERVATION → OUTPUT
```

**Example Reasoning:**
```
[Input] Patient turn: "I'm really scared about what this means for my family"

[Thought] This is a patient turn. I need to identify Empathic Opportunities.
[Action] Scan for emotional language
[Observation] Found "really scared" - explicit emotional expression

[Thought] "scared" is a direct feeling expression, not implied
[Action] Classify as explicit_feeling
[Observation] Confidence: 0.92

[Output]
- Span: "really scared" → explicit_feeling (0.92)
```

**Pros:**
- Transparent reasoning
- Follows the academic framework explicitly
- Explainable decisions

**Cons:**
- More LLM calls for complex turns
- Slightly slower than direct classification

### 2. Multi-Agent System

A coordinated system of specialized agents:

| Agent | Role | Handles |
|-------|------|---------|
| **EO Detector** | Find patient Empathic Opportunities | Patient turns only |
| **Response Classifier** | Classify clinician responses/elicitations | Clinician turns only |
| **SPIKES Tagger** | Assign SPIKES protocol stages | Clinician turns only |
| **Relation Linker** | Connect EOs to clinician responses | All conversations |

**Workflow:**
```
                    ┌─────────────────┐
                    │   Coordinator   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   EO Detector   │ │Response Classif.│ │  SPIKES Tagger  │
│  (Patient only) │ │(Clinician only) │ │(Clinician only) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relation Linker │
                    └─────────────────┘
```

**Pros:**
- Modular - each agent can be optimized independently
- Parallel processing for clinician turns
- Specialized prompts for each task

**Cons:**
- More complex coordination
- Higher total API cost

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.18.0
pypdf>=3.17.0
openai>=1.0.0
anthropic>=0.18.0
```

### API Keys

Set your API key as an environment variable:

```bash
# For OpenAI
export OPENAI_API_KEY=your-openai-api-key

# For Anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

Or configure it in the UI sidebar.

---

## Configuration

### Agent Configuration Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `provider` | `openai`, `anthropic` | `openai` | LLM provider |
| `model` | See below | `gpt-4o` | Model to use |
| `agent_type` | `react`, `multi_agent` | `react` | Agent architecture |
| `temperature` | 0.0 - 1.0 | 0.1 | LLM temperature |

### Available Models

**OpenAI:**
- `gpt-4o` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307`

---

## Usage

### Using the UI

1. **Start the application:**
   ```bash
   streamlit run app/main.py
   ```

2. **Configure the agent** in the sidebar under "Agent Configuration"

3. **Use Agent Mode (Full Conversation):**
   - Select "🤖 Agent" mode in sidebar
   - Click "Run Agent on Full Conversation"
   - View results or compare with expert annotations
   - Accept/Export/Clear as needed

4. **Use AI Suggestions (Per Turn):**
   - Select "✏️ Annotate" mode
   - Select a turn to annotate
   - Click "Generate AI Suggestions" in the right panel
   - Accept, Modify, or Reject each suggestion

### Using the Python API

```python
from agents import ReActAnnotationAgent, MultiAgentSystem
from agents.runner import AnnotationRunner, annotate_file

# Option 1: Using AnnotationRunner (recommended)
runner = AnnotationRunner(
    agent_type="react",      # or "multi_agent"
    provider="openai",       # or "anthropic"
    model="gpt-4o",
    api_key="your-api-key"   # optional if env var set
)

# Annotate a file
result = runner.annotate_file("data/samples/sample_conversation_1.json")

# Save results
runner.save_result(result, "output/annotated.json")

# Compare with expert
metrics = runner.compare_with_expert(result, "expert_annotations.json")
print(f"F1 Score: {metrics['f1']:.2f}")
```

```python
# Option 2: Using agents directly
from agents.base import AgentConfig

config = AgentConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.1
)

# ReAct Agent
agent = ReActAnnotationAgent(config)
result = agent.annotate_conversation(conversation)

# Multi-Agent System
system = MultiAgentSystem(config)
result = system.annotate_conversation(conversation, parallel=True)
```

### Using the CLI

```bash
# Basic usage
python -m agents.runner data/samples/sample_conversation_1.json

# With options
python -m agents.runner input.json \
    -t multi_agent \
    -p anthropic \
    -m claude-3-5-sonnet-20241022 \
    -o output.json

# Skip relation detection
python -m agents.runner input.json --no-relations
```

---

## API Reference

### AnnotationResult

The output of agent annotation:

```python
@dataclass
class AnnotationResult:
    conversation_id: str
    turns: List[TurnAnnotation]
    agent_type: str  # 'react' or 'multi_agent'
    metadata: Dict
```

### TurnAnnotation

Annotation for a single turn:

```python
@dataclass
class TurnAnnotation:
    turn_id: int
    speaker: Speaker  # PATIENT or CLINICIAN
    text: str
    spikes_stage: Optional[str]
    spans: List[SpanAnnotation]
    relations: List[RelationAnnotation]
```

### SpanAnnotation

A single span annotation:

```python
@dataclass
class SpanAnnotation:
    span_id: str
    text: str
    start: int
    end: int
    label: str
    reasoning: str
```

### Output JSON Format

```json
{
  "conversation_id": "conv_001",
  "agent_type": "react",
  "metadata": {
    "model": "gpt-4o",
    "provider": "openai"
  },
  "turns": [
    {
      "turn_id": 1,
      "speaker": "patient",
      "text": "I'm really scared about the results.",
      "annotations": {
        "spikes_stage": null,
        "spans": [
          {
            "span_id": "span_t1_0",
            "text": "really scared",
            "start": 4,
            "end": 17,
            "label": "explicit_feeling",
            "reasoning": "Direct expression of fear emotion"
          }
        ],
        "relations": []
      }
    }
  ]
}
```

---

## Training Workflow

The system is designed for iterative improvement through expert feedback:

### Phase 1: Initial Annotation

```
1. Agent annotates conversation
2. Results stored with source='agent'
```

### Phase 2: Expert Review

```
1. Expert reviews agent suggestions
2. Accepts, modifies, or rejects each annotation
3. Adds missed annotations
4. Results stored with source='expert'
```

### Phase 3: Comparison & Training

```
1. Compare agent vs expert annotations
2. Calculate metrics (precision, recall, F1, span IoU)
3. Identify patterns in agent errors
4. Retrain/fine-tune agent (future feature)
```

### Comparison Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Agent annotations that match expert |
| **Recall** | Expert annotations found by agent |
| **F1** | Harmonic mean of precision and recall |
| **Span IoU** | Intersection over Union of span boundaries |
| **Missed** | Annotations expert added that agent missed |
| **False Positives** | Agent annotations rejected by expert |

---

## Label Reference

### Patient Labels (Empathic Opportunities)

| Label | Description | Example |
|-------|-------------|---------|
| `explicit_feeling` | Direct emotion expression | "I'm scared" |
| `implicit_feeling` | Indirect emotional reference | "My aunt died from this" |
| `explicit_appreciation` | Direct attitude toward things/events | "The MRI was terrible" |
| `implicit_appreciation` | Indirect attitude | "My symptoms aren't improving" |
| `explicit_judgement` | Direct attitude toward self/others | "I'm bad at taking pills" |
| `implicit_judgement` | Indirect judgement | "I forget my medication" |

### Clinician Labels

**Elicitations:**
| Label | Description |
|-------|-------------|
| `direct_elicitation_feeling` | "How do you feel about that?" |
| `indirect_elicitation_feeling` | "So you're worried that..." |
| `direct_elicitation_appreciation` | "How was the MRI for you?" |
| `indirect_elicitation_appreciation` | "It sounds like the medication isn't helping" |
| `direct_elicitation_judgement` | "Do you think you're handling this well?" |
| `indirect_elicitation_judgement` | "So the nurse wasn't helpful?" |

**Responses:**
| Label | Description |
|-------|-------------|
| `acceptance_positive_regard_*` | Validation, positive judgement |
| `acceptance_neutral_support_*` | Normalizing, denying negative self-assessment |
| `sharing_feeling/appreciation/judgement` | Expressed agreement |
| `understanding_feeling/appreciation/judgement` | Acknowledgement |

### SPIKES Stages

| Stage | Description |
|-------|-------------|
| `setting` | Establishing environment |
| `perception` | Understanding patient's knowledge |
| `invitation` | Assessing information preferences |
| `knowledge` | Delivering medical information |
| `empathy` | Responding to emotions |
| `strategy` | Planning next steps |

---

## Troubleshooting

### Common Issues

**"Failed to import agents"**
- Ensure you're running from the project root directory
- Check that `agents/` directory exists with all files

**"API key not set"**
- Set environment variable: `export OPENAI_API_KEY=...`
- Or enter key in sidebar configuration

**Agent returns empty results**
- Check API key is valid
- Verify model name is correct

**Slow performance**
- Use `gpt-4o-mini` or `claude-3-haiku` for faster responses
- Use `react` agent (fewer API calls than `multi_agent`)
- Disable relation detection with `include_relations=False`

---

## Future Enhancements

- [ ] Fine-tuning support for custom models
- [ ] Active learning for annotation prioritization
- [ ] Inter-annotator agreement metrics
- [ ] Batch processing for multiple conversations
- [ ] Database storage for annotations
- [ ] Model evaluation dashboard

---

## References

- Lahnala et al. (2024): "Appraisal Framework for Clinical Empathy: A Novel Application to Breaking Bad News Conversations" (LREC-COLING 2024)
- SPIKES Protocol: Baile et al. (2000)
- Pounds' Appraisal Framework for Clinical Empathy
