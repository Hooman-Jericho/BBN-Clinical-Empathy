"""
Agent Runner

Unified interface for running BBN annotation agents.
Supports both ReAct and Multi-Agent systems.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List
from datetime import datetime

from .base import AgentConfig, AnnotationResult
from .react_agent import ReActAnnotationAgent
from .multi_agent import MultiAgentSystem


AgentType = Literal["react", "multi_agent"]


class AnnotationRunner:
    """
    Unified runner for BBN annotation agents.

    Provides a simple interface to:
    1. Load conversations from files
    2. Run annotation with either agent type
    3. Save results
    4. Compare agent vs expert annotations
    """

    def __init__(
        self,
        agent_type: AgentType = "react",
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **config_kwargs
    ):
        """
        Initialize the runner.

        Args:
            agent_type: "react" or "multi_agent"
            provider: "openai" or "anthropic"
            model: Model name (defaults based on provider)
            api_key: API key (uses env var if not provided)
            **config_kwargs: Additional AgentConfig parameters
        """
        normalized_agent_type = "multi_agent" if agent_type in ("multi", "multi_agent") else "react"
        self.agent_type = normalized_agent_type

        # Set default model
        if model is None:
            model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"

        # Create config
        self.config = AgentConfig(
            provider=provider,
            model=model,
            api_key=api_key or os.getenv(
                "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
            ),
            **config_kwargs
        )

        # Initialize appropriate agent
        if agent_type == "react":
            self.agent = ReActAnnotationAgent(self.config)
        else:
            self.agent = MultiAgentSystem(self.config)

    def annotate_file(
        self,
        file_path: str,
        include_relations: bool = True
    ) -> AnnotationResult:
        """
        Annotate a conversation from a JSON file.

        Args:
            file_path: Path to conversation JSON file
            include_relations: Whether to identify relations

        Returns:
            AnnotationResult with annotations
        """
        with open(file_path, 'r') as f:
            conversation = json.load(f)

        return self.annotate(conversation, include_relations)

    def annotate(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True
    ) -> AnnotationResult:
        """
        Annotate a conversation.

        Args:
            conversation: Conversation dict
            include_relations: Whether to identify relations

        Returns:
            AnnotationResult with annotations
        """
        return self.agent.annotate_conversation(conversation, include_relations)

    def annotate_turn(
        self,
        turn: Dict[str, Any],
        context: str = ""
    ):
        """
        Annotate a single turn (for real-time suggestions).

        Args:
            turn: Turn dict with turn_id, speaker, text
            context: Previous conversation context

        Returns:
            TurnAnnotation
        """
        if self.agent_type == "react":
            return self.agent.annotate_turn(turn, context)
        else:
            # Multi-agent processes whole conversation, so wrap the turn
            return self.agent._process_turn(turn, context, parallel=False)

    def save_result(
        self,
        result: AnnotationResult,
        output_path: str
    ):
        """Save annotation result to a JSON file."""
        output_data = result.to_dict()
        output_data["metadata"]["annotated_at"] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def compare_with_expert(
        self,
        agent_result: AnnotationResult,
        expert_file: str
    ) -> Dict[str, Any]:
        """
        Compare agent annotations with expert annotations.

        Args:
            agent_result: Agent's annotation result
            expert_file: Path to expert-annotated JSON file

        Returns:
            Comparison metrics
        """
        with open(expert_file, 'r') as f:
            expert_data = json.load(f)

        metrics = {
            "total_turns": len(agent_result.turns),
            "label_matches": 0,
            "label_mismatches": 0,
            "agent_only": 0,  # Agent found, expert didn't
            "expert_only": 0,  # Expert found, agent didn't
            "span_overlap_scores": [],
            "per_turn_details": [],
        }

        expert_turns = {t["turn_id"]: t for t in expert_data.get("turns", [])}

        for agent_turn in agent_result.turns:
            turn_id = agent_turn.turn_id
            expert_turn = expert_turns.get(turn_id, {})
            expert_spans = expert_turn.get("annotations", {}).get("spans", [])

            agent_labels = {s.label for s in agent_turn.spans}
            expert_labels = {s.get("label") for s in expert_spans}

            turn_detail = {
                "turn_id": turn_id,
                "agent_labels": list(agent_labels),
                "expert_labels": list(expert_labels),
                "matches": list(agent_labels & expert_labels),
                "agent_only": list(agent_labels - expert_labels),
                "expert_only": list(expert_labels - agent_labels),
            }

            metrics["label_matches"] += len(turn_detail["matches"])
            metrics["agent_only"] += len(turn_detail["agent_only"])
            metrics["expert_only"] += len(turn_detail["expert_only"])
            metrics["per_turn_details"].append(turn_detail)

            # Calculate span overlap (IoU) for matching labels
            for agent_span in agent_turn.spans:
                for expert_span in expert_spans:
                    if agent_span.label == expert_span.get("label"):
                        iou = self._calculate_span_iou(
                            (agent_span.start, agent_span.end),
                            (expert_span.get("start", 0), expert_span.get("end", 0))
                        )
                        metrics["span_overlap_scores"].append(iou)

        # Calculate summary metrics
        total_agent = sum(len(t.spans) for t in agent_result.turns)
        total_expert = sum(
            len(expert_turns.get(t.turn_id, {}).get("annotations", {}).get("spans", []))
            for t in agent_result.turns
        )

        if total_agent + total_expert > 0:
            metrics["precision"] = metrics["label_matches"] / max(total_agent, 1)
            metrics["recall"] = metrics["label_matches"] / max(total_expert, 1)
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1"] = (
                    2 * metrics["precision"] * metrics["recall"] /
                    (metrics["precision"] + metrics["recall"])
                )
            else:
                metrics["f1"] = 0.0
        else:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0

        if metrics["span_overlap_scores"]:
            metrics["avg_span_iou"] = sum(metrics["span_overlap_scores"]) / len(metrics["span_overlap_scores"])
        else:
            metrics["avg_span_iou"] = 0.0

        return metrics

    def _calculate_span_iou(
        self,
        span1: tuple,
        span2: tuple
    ) -> float:
        """Calculate Intersection over Union for two spans."""
        start1, end1 = span1
        start2, end2 = span2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)

        union = (end1 - start1) + (end2 - start2) - intersection

        if union == 0:
            return 0.0
        return intersection / union


def annotate_conversation(
    conversation: Dict[str, Any],
    agent_type: AgentType = "react",
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> AnnotationResult:
    """
    Convenience function to annotate a conversation.

    Args:
        conversation: Conversation dict
        agent_type: "react" or "multi_agent"
        provider: "openai" or "anthropic"
        model: Model name
        api_key: API key
        **kwargs: Additional config parameters

    Returns:
        AnnotationResult

    Example:
        >>> result = annotate_conversation(
        ...     conversation,
        ...     agent_type="react",
        ...     provider="openai"
        ... )
    """
    runner = AnnotationRunner(
        agent_type=agent_type,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return runner.annotate(conversation)


def annotate_file(
    file_path: str,
    agent_type: AgentType = "react",
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> AnnotationResult:
    """
    Convenience function to annotate a conversation file.

    Args:
        file_path: Path to conversation JSON
        agent_type: "react" or "multi_agent"
        provider: "openai" or "anthropic"
        model: Model name
        api_key: API key
        output_path: Optional path to save results
        **kwargs: Additional config parameters

    Returns:
        AnnotationResult

    Example:
        >>> result = annotate_file(
        ...     "data/samples/sample_conversation_1.json",
        ...     agent_type="multi_agent",
        ...     provider="anthropic",
        ...     output_path="output/annotated.json"
        ... )
    """
    runner = AnnotationRunner(
        agent_type=agent_type,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    result = runner.annotate_file(file_path)

    if output_path:
        runner.save_result(result, output_path)

    return result


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BBN Annotation Agent Runner")
    parser.add_argument("input", help="Input conversation JSON file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-t", "--type",
        choices=["react", "multi_agent"],
        default="react",
        help="Agent type (default: react)"
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--no-relations", action="store_true", help="Skip relation detection")

    args = parser.parse_args()

    # Run annotation
    runner = AnnotationRunner(
        agent_type=args.type,
        provider=args.provider,
        model=args.model,
    )

    print(f"Annotating {args.input} with {args.type} agent ({args.provider})...")
    result = runner.annotate_file(args.input, include_relations=not args.no_relations)

    # Output
    if args.output:
        runner.save_result(result, args.output)
        print(f"Results saved to {args.output}")
    else:
        print(result.to_json())
