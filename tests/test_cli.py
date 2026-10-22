import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.cli import main

SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "sample_conversation_1.json"


def test_validate_command(capsys):
    assert main(["validate", str(SAMPLE)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_command_fails_on_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    data = json.loads(SAMPLE.read_text())
    data["turns"][2]["annotations"]["spans"][0]["start"] += 1
    bad.write_text(json.dumps(data))
    assert main(["validate", str(bad)]) == 1


def test_gold_vs_itself_is_perfect(tmp_path):
    out = tmp_path / "r.json"
    # a gold file converted to a prediction file by the data model must score F1 = 1
    from agents.base import AnnotationResult
    gold = json.loads(SAMPLE.read_text())
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(AnnotationResult.from_dict(gold).to_dict()))
    assert main(["evaluate", "--pred", str(pred), "--gold", str(SAMPLE), "--n-boot", "20",
                 "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["headline"]["span_f1@exact"] == 1.0
    assert rep["headline"]["spikes_accuracy"] == 1.0
    assert rep["headline"]["relation_f1"] == 1.0


def test_legacy_invocation_maps_to_annotate(monkeypatch):
    import agents.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "cmd_annotate", lambda a: seen.setdefault("a", a) and 0)
    ap = cli.build_parser()
    monkeypatch.setitem(ap._subparsers._group_actions[0].choices["annotate"]._defaults, "func",
                        cli.cmd_annotate)
    assert cli.main(["x.json", "--agent", "react"]) == 0
    assert seen["a"].input == "x.json"


def test_experiment_end_to_end_with_stub_llm(tmp_path, monkeypatch):
    """Whole pipeline (strip gold -> agent -> evaluate -> aggregate) with a stubbed model."""
    from agents.llm import LLMClient
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    monkeypatch.setattr(LLMClient, "chat", lambda self, *a, **k: json.dumps(
        {"thought": "x", "action": "finish", "action_input": {"annotations": []}}))
    out = tmp_path / "exp.json"
    assert main(["experiment", "--gold", str(SAMPLE), "--runs", "2", "--n-boot", "10",
                 "--agent", "react", "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["n_runs"] == 2 and rep["config"]["api_key"] is None
    assert rep["aggregate"]["span_f1@iou>=0.5"]["mean"] == 0.0  # empty predictions
