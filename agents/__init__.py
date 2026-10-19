"""
BBN Annotation Agents

Two agent implementations for annotating Breaking Bad News conversations:
1. ReAct Agent - Single agent with step-by-step reasoning
2. Multi-Agent System - Specialized agents for different annotation tasks
"""

from .react_agent import ReActAnnotationAgent
from .multi_agent import MultiAgentSystem
from .base import AnnotationResult, AgentConfig

__all__ = [
    "ReActAnnotationAgent",
    "MultiAgentSystem",
    "AnnotationResult",
    "AgentConfig",
]
