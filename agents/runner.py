"""Agent Runner with mathematically rigorous Bipartite Span Matching evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List, Tuple
from datetime import datetime

from .base import AgentConfig, AnnotationResult, TurnAnnotation, SpanAnnotation
from .react_agent import ReActAnnotationAgent
from .multi_agent import MultiAgentSystem
from .evaluation import solve_max_weight_matching, span_iou

AgentType = Literal["react", "multi_agent"]


class AnnotationRunner:
    """Unified runner for BBN annotation agents with exact bipartite evaluation metrics."""

    def __init__(
        self,
        agent_type: AgentType = "react",
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_critic: bool = True,
        use_coordinator: bool = False,
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
            self.agent = MultiAgentSystem(
                self.config, use_critic=use_critic, use_coordinator=use_coordinator)

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

    @staticmethod
    def _solve_max_weight_bipartite_matching(weights: List[List[float]]) -> List[Tuple[int, int]]:
        """Globally optimal 1-to-1 matching (delegates to :mod:`agents.evaluation`)."""
        return solve_max_weight_matching(weights)

    def compare_with_expert(
        self,
        agent_result: AnnotationResult,
        expert_file: Any,
        min_iou_threshold: float = 0.5,
        exact_match: bool = False,
    ) -> Dict[str, Any]:
        """Compare agent annotations with expert ground-truth using 1-to-1 Maximum Bipartite Matching.
        
        Rigorous scientific evaluation protocol:
        - Matches individual spans via Maximum Weight Bipartite Matching (Hungarian algorithm).
        - Prevents one prediction from matching multiple expert annotations and vice-versa.
        - Supports exact character offset matching or IoU threshold matching.
        - Handles zero denominators safely without NaN or infinite values.
        - Resolves ambiguous multi-span assignments globally without greedy order bias.
        """
        if isinstance(expert_file, (str, Path)):
            with open(expert_file, 'r', encoding='utf-8') as f:
                expert_data = json.load(f)
        elif isinstance(expert_file, dict):
            expert_data = expert_file
        elif hasattr(expert_file, "to_dict"):
            expert_data = expert_file.to_dict()
        else:
            raise TypeError("expert_file must be a filepath string, Path, dict, or AnnotationResult")

        expert_turns = {
            t["turn_id"]: t for t in expert_data.get("turns", []) if isinstance(t, dict) and "turn_id" in t
        }
        agent_turns_map = {t.turn_id: t for t in agent_result.turns}

        all_turn_ids = sorted(set(agent_turns_map.keys()) | set(expert_turns.keys()))

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious: List[float] = []
        per_turn_details: List[Dict[str, Any]] = []

        CARDINAL_WEIGHT = 1000.0  # Guarantees maximizing number of matches before maximizing IoU

        for turn_id in all_turn_ids:
            agent_turn = agent_turns_map.get(turn_id)
            expert_turn = expert_turns.get(turn_id, {})
            agent_spans = agent_turn.spans if agent_turn else []
            expert_spans = expert_turn.get("annotations", {}).get("spans", []) if isinstance(expert_turn, dict) else []

            n_a = len(agent_spans)
            n_e = len(expert_spans)

            if n_a == 0 or n_e == 0:
                tp = 0
                fp = n_a
                fn = n_e
                per_turn_details.append({
                    "turn_id": turn_id,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "matched_pairs": [],
                    "agent_only": [s.to_dict() if hasattr(s, "to_dict") else s for s in agent_spans],
                    "expert_only": list(expert_spans),
                })
                total_tp += tp
                total_fp += fp
                total_fn += fn
                continue

            # Build weight matrix for Maximum Bipartite Matching
            weight_matrix = [[0.0] * n_e for _ in range(n_a)]
            iou_matrix = [[0.0] * n_e for _ in range(n_a)]

            for i, a_span in enumerate(agent_spans):
                for j, e_span in enumerate(expert_spans):
                    e_label = e_span.get("label", "")
                    if a_span.label == e_label:
                        a_start, a_end = a_span.start, a_span.end
                        e_start = e_span.get("start", e_span.get("start_pos", 0))
                        e_end = e_span.get("end", e_span.get("end_pos", 0))

                        iou = self._calculate_span_iou((a_start, a_end), (e_start, e_end))
                        iou_matrix[i][j] = iou

                        if exact_match:
                            is_valid = (a_start == e_start and a_end == e_end)
                        else:
                            is_valid = iou > 0.0 and iou >= min_iou_threshold

                        if is_valid:
                            weight_matrix[i][j] = CARDINAL_WEIGHT + iou

            # Solve globally optimal assignment
            matched_indices = self._solve_max_weight_bipartite_matching(weight_matrix)

            matched_a_set = set()
            matched_e_set = set()
            matched_pairs_info = []

            for a_idx, e_idx in matched_indices:
                matched_a_set.add(a_idx)
                matched_e_set.add(e_idx)
                iou_val = iou_matrix[a_idx][e_idx]
                matched_ious.append(iou_val)
                matched_pairs_info.append({
                    "agent_span": agent_spans[a_idx].to_dict() if hasattr(agent_spans[a_idx], "to_dict") else agent_spans[a_idx],
                    "expert_span": expert_spans[e_idx],
                    "iou": iou_val,
                })

            tp = len(matched_indices)
            fp = n_a - tp
            fn = n_e - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn

            per_turn_details.append({
                "turn_id": turn_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matched_pairs": matched_pairs_info,
                "agent_only": [agent_spans[i].to_dict() if hasattr(agent_spans[i], "to_dict") else agent_spans[i] for i in range(n_a) if i not in matched_a_set],
                "expert_only": [expert_spans[j] for j in range(n_e) if j not in matched_e_set],
            })

        # Summary Metrics with safe zero-denominator handling
        precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        avg_span_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

        return {
            "total_turns": len(all_turn_ids),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "label_matches": total_tp,  # Backward compatible alias
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_span_iou": round(avg_span_iou, 4),
            "span_overlap_scores": matched_ious,
            "per_turn_details": per_turn_details,
        }

    @staticmethod
    def _calculate_span_iou(
        span1: Tuple[int, int],
        span2: Tuple[int, int]
    ) -> float:
        """Calculate Intersection over Union for two character ranges."""
        return span_iou(span1, span2)


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


if __name__ == "__main__":  # pragma: no cover
    from .cli import main

    raise SystemExit(main())
