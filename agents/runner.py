"""Agent Runner with mathematically rigorous Bipartite Span Matching evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List, Tuple
from datetime import datetime

from .base import AgentConfig, AnnotationResult, TurnAnnotation, SpanAnnotation
from .react_agent import ReActAnnotationAgent
from .multi_agent import MultiAgentSystem

AgentType = Literal["react", "multi_agent"]


class AnnotationRunner:
    """Unified runner for BBN annotation agents with exact bipartite evaluation metrics."""

    def __init__(
        self,
        agent_type: AgentType = "react",
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **config_kwargs
    ):
        normalized_agent_type = "multi_agent" if agent_type in ("multi", "multi_agent") else "react"
        self.agent_type = normalized_agent_type

        if model is None:
            model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"

        self.config = AgentConfig(
            provider=provider,
            model=model,
            api_key=api_key or os.getenv(
                "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
            ),
            **config_kwargs
        )

        if self.agent_type == "react":
            self.agent = ReActAnnotationAgent(self.config)
        else:
            self.agent = MultiAgentSystem(self.config)

    def annotate_file(
        self,
        file_path: str,
        include_relations: bool = True
    ) -> AnnotationResult:
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        return self.annotate(conversation, include_relations)

    def annotate(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True
    ) -> AnnotationResult:
        return self.agent.annotate_conversation(conversation, include_relations)

    def annotate_turn(
        self,
        turn: Dict[str, Any],
        context: str = ""
    ) -> TurnAnnotation:
        if self.agent_type == "react":
            return self.agent.annotate_turn(turn, context)
        else:
            return self.agent._process_turn(turn, context, parallel=False)

    def save_result(
        self,
        result: AnnotationResult,
        output_path: str
    ):
        output_data = result.to_dict()
        if "metadata" not in output_data:
            output_data["metadata"] = {}
        output_data["metadata"]["annotated_at"] = datetime.now().isoformat()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def compare_with_expert(
        self,
        agent_result: AnnotationResult,
        expert_file: str,
        min_iou_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Compare agent annotations with expert ground-truth using 1-to-1 Bipartite Span Matching.
        
        Solves the precision compression and Cartesian product bugs:
        - Evaluates at the individual span level (not set of labels).
        - Matches each agent span to at most one expert span with maximal IoU.
        - Calculates True Positives (TP), False Positives (FP), and False Negatives (FN).
        """
        with open(expert_file, 'r', encoding='utf-8') as f:
            expert_data = json.load(f)

        expert_turns = {t["turn_id"]: t for t in expert_data.get("turns", [])}

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious: List[float] = []
        per_turn_details: List[Dict[str, Any]] = []

        for agent_turn in agent_result.turns:
            turn_id = agent_turn.turn_id
            expert_turn = expert_turns.get(turn_id, {})
            expert_spans = expert_turn.get("annotations", {}).get("spans", [])

            # Generate candidate pairs where label matches exactly
            candidates: List[Tuple[float, int, int]] = []
            for i, a_span in enumerate(agent_turn.spans):
                for j, e_span in enumerate(expert_spans):
                    if a_span.label == e_span.get("label"):
                        iou = self._calculate_span_iou(
                            (a_span.start, a_span.end),
                            (e_span.get("start", 0), e_span.get("end", 0))
                        )
                        if iou > min_iou_threshold:
                            candidates.append((iou, i, j))

            # Optimal greedy 1-to-1 bipartite matching
            candidates.sort(key=lambda x: x[0], reverse=True)
            matched_a_indices = set()
            matched_e_indices = set()
            turn_ious = []
            matched_pairs_info = []

            for iou, a_idx, e_idx in candidates:
                if a_idx not in matched_a_indices and e_idx not in matched_e_indices:
                    matched_a_indices.add(a_idx)
                    matched_e_indices.add(e_idx)
                    turn_ious.append(iou)
                    matched_pairs_info.append({
                        "agent_span": agent_turn.spans[a_idx].to_dict(),
                        "expert_span": expert_spans[e_idx],
                        "iou": iou,
                    })

            tp = len(matched_a_indices)
            fp = len(agent_turn.spans) - tp
            fn = len(expert_spans) - len(matched_e_indices)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            matched_ious.extend(turn_ious)

            per_turn_details.append({
                "turn_id": turn_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matched_pairs": matched_pairs_info,
                "agent_only": [agent_turn.spans[i].to_dict() for i in range(len(agent_turn.spans)) if i not in matched_a_indices],
                "expert_only": [expert_spans[j] for j in range(len(expert_spans)) if j not in matched_e_indices],
            })

        # Summary Metrics
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        avg_span_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

        return {
            "total_turns": len(agent_result.turns),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "label_matches": total_tp,  # Backward compatible alias
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_span_iou": avg_span_iou,
            "span_overlap_scores": matched_ious,
            "per_turn_details": per_turn_details,
        }

    def _calculate_span_iou(
        self,
        span1: tuple,
        span2: tuple
    ) -> float:
        """Calculate Intersection over Union for two character ranges."""
        start1, end1 = span1
        start2, end2 = span2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)

        union = (end1 - start1) + (end2 - start2) - intersection
        if union <= 0:
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
