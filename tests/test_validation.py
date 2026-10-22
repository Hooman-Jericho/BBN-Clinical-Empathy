"""Unit tests for strict output validation, the LLM client and agent robustness."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import AgentConfig, Speaker
from agents.llm import LLMClient
from agents.multi_agent import MultiAgentSystem
from agents.react_agent import ReActAnnotationAgent, build_react_system_prompt
from agents.validation import (ParseStats, build_relations, build_spans, parse_turn_payload,
                               select_critic_spans)

TEXT = "I am so scared about what this means for my family."


def _spans(raw, speaker=Speaker.PATIENT, text=TEXT):
    stats = ParseStats()
    return build_spans(raw, 1, text, speaker, stats), stats


def test_bad_span_does_not_drop_siblings():
    out, st = _spans([
        {"text": "so scared", "label": "explicit_feeling"},
        {"text": "xx", "label": "explicit_feeling", "start": "abc"},  # text not in turn
        "garbage",
        {"text": "my family", "label": "implicit_feeling"},
    ])
    assert [s.text for s in out] == ["so scared", "my family"]
    assert st.spans_proposed == 4 and st.spans_accepted == 2
    assert st.spans_rejected_text == 1 and st.spans_rejected_schema == 1


def test_invented_and_wrong_speaker_labels_rejected():
    out, st = _spans([
        {"text": "so scared", "label": "made_up_label"},
        {"text": "so scared", "label": "direct_elicitation_feeling"},  # clinician label
    ])
    assert out == []
    assert st.spans_rejected_label == 1 and st.spans_rejected_speaker == 1


def test_confidence_not_fabricated_and_duplicates_removed():
    out, st = _spans([
        {"text": "so scared", "label": "explicit_feeling"},
        {"text": "so scared", "label": "explicit_feeling", "confidence": 0.7},
        {"text": "my family", "label": "implicit_feeling", "confidence": 7},
    ])
    assert len(out) == 2 and st.spans_rejected_duplicate == 1
    assert out[0].confidence is None
    assert out[1].confidence is None  # out-of-range -> None, not clipped silently


def test_wrong_offsets_are_corrected_and_counted():
    out, st = _spans([{"text": "my family", "label": "implicit_feeling", "start": 3, "end": 12}])
    assert TEXT[out[0].start:out[0].end] == "my family"
    assert st.spans_boundary_corrected == 1
    _, st2 = _spans([{"text": "my family", "label": "implicit_feeling"}])
    assert st2.spans_boundary_corrected == 0  # no hint given => nothing was "corrected"


def test_parse_payload_status():
    st = ParseStats()
    assert parse_turn_payload({}, Speaker.PATIENT, st)[2] == "parse_failed"
    assert parse_turn_payload({"annotations": "x"}, Speaker.PATIENT, st)[2] == "schema_failed"
    stage, raw, status = parse_turn_payload(
        {"annotations": [], "spikes_stage": "bogus"}, Speaker.CLINICIAN, st)
    assert (stage, raw, status) == (None, [], "ok") and st.spikes_invalid == 1


def test_relation_validation():
    st = ParseStats()
    parsed = {"relations": [
        {"from_span_id": "el_1", "to_span_id": "eo_1", "relation_type": "elicitation_of"},
        {"from_span_id": "eo_1", "to_span_id": "el_1", "relation_type": "response_to"},  # reversed
        {"from_span_id": "el_9", "to_span_id": "eo_1", "relation_type": "response_to"},  # unknown
        {"from_span_id": "el_1", "to_span_id": "eo_1", "relation_type": "elicitation_of"},  # dup
    ]}
    rels = build_relations(parsed, {"eo_1"}, {"el_1"}, st)
    assert len(rels) == 1 and st.relations_accepted == 1 and st.relations_rejected == 3


def test_critic_selection_semantics():
    props = [{"text": "so scared", "label": "explicit_feeling", "start": 5, "end": 14}]
    st = ParseStats()
    assert select_critic_spans({"approved_spans": []}, props, st) == []  # may reject all
    assert st.critic_rejected_all == 1
    assert select_critic_spans("junk", props, st) is None and st.critic_fallbacks == 1
    inv = {"approved_spans": [{"text": "zzz", "label": "explicit_feeling", "start": 40, "end": 43}]}
    assert select_critic_spans(inv, props, st) == [] and st.critic_invented == 1


# ----------------------------------------------------------------------------- agents
def _cfg(**kw):
    return AgentConfig(provider="openai", model="m", api_key="sk-x", **kw)


def test_react_prompt_lists_labels():
    p = build_react_system_prompt(Speaker.PATIENT)
    assert "explicit_feeling" in p and "direct_elicitation_feeling" not in p
    c = build_react_system_prompt(Speaker.CLINICIAN)
    assert "direct_elicitation_feeling" in c and "perception" in c


def test_react_unparseable_step_is_not_silent():
    agent = ReActAnnotationAgent(_cfg(), max_steps=2)
    agent.llm.chat = MagicMock(return_value="I refuse to answer in JSON")
    res = agent.annotate_turn({"turn_id": 1, "speaker": "patient", "text": TEXT})
    assert res.spans == []
    assert res.diagnostics  # failure is recorded, not hidden


def test_react_good_run_has_clean_diagnostics():
    agent = ReActAnnotationAgent(_cfg(), max_steps=2)
    agent.llm.chat = MagicMock(return_value=json.dumps({
        "thought": "done", "action": "finish",
        "action_input": {"annotations": [{"text": "so scared", "label": "explicit_feeling"}]}}))
    res = agent.annotate_turn({"turn_id": 1, "speaker": "patient", "text": TEXT})
    assert [s.text for s in res.spans] == ["so scared"]


def _system(responses, **kw):
    sys_ = MultiAgentSystem(_cfg(), **kw)
    it = iter(responses)
    sys_.llm.chat = MagicMock(side_effect=lambda *a, **k: next(it))
    return sys_


PROPOSAL = json.dumps({"annotations": [{"text": "so scared", "label": "explicit_feeling"}]})


def test_critic_can_reject_everything():
    s = _system([PROPOSAL, json.dumps({"approved_spans": []})])
    res = s._process_turn({"turn_id": 1, "speaker": "patient", "text": TEXT}, "", parallel=False)
    assert res.spans == []


def test_critic_failure_falls_back_and_is_flagged():
    s = _system([PROPOSAL, "not json"])
    res = s._process_turn({"turn_id": 1, "speaker": "patient", "text": TEXT}, "", parallel=False)
    assert len(res.spans) == 1
    assert res.diagnostics["stats"].get("critic_fallbacks") == 1


def test_coordinator_off_by_default_and_used_when_on():
    s = _system([PROPOSAL, PROPOSAL], use_critic=False)
    s._process_turn({"turn_id": 1, "speaker": "patient", "text": TEXT}, "", parallel=False)
    assert s.llm.chat.call_count == 1  # specialist only
    plan = json.dumps({"speaker_role": "patient", "active_specialists": ["eo"],
                       "coordination_strategy": "LOOK-FOR-FEAR"})
    s = _system([plan, PROPOSAL], use_critic=False, use_coordinator=True)
    s._process_turn({"turn_id": 1, "speaker": "patient", "text": TEXT}, "", parallel=False)
    assert "LOOK-FOR-FEAR" in s.llm.chat.call_args_list[1].args[1]


# ----------------------------------------------------------------------------- llm
def test_llm_request_kwargs():
    c = LLMClient(_cfg(seed=7, json_mode=True, temperature=0.3))
    k = c._openai_kwargs("s", "u")
    assert k["seed"] == 7 and k["response_format"] == {"type": "json_object"}
    assert k["temperature"] == 0.3
    assert "seed" not in LLMClient(_cfg())._openai_kwargs("s", "u")
    a = LLMClient(AgentConfig(provider="anthropic", model="m", api_key="k", temperature=0.4))
    assert a._anthropic_kwargs("s", "u")["temperature"] == 0.4


def test_llm_usage_counters():
    c = LLMClient(_cfg())
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    resp.usage.prompt_tokens, resp.usage.completion_tokens = 10, 5
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    with patch.object(c, "_get_client", return_value=client):
        c.chat("s", "u")
        c.chat("s", "u")
    assert c.usage() == {"calls": 2, "input_tokens": 20, "output_tokens": 10}
