"""Command line interface: ``python -m agents <command>``.

Commands
--------
validate    Check conversation/gold JSON files (offsets, labels, relations).
annotate    Annotate one conversation with an agent (needs an API key).
evaluate    Score prediction files against gold files (no API key needed).
experiment  Annotate gold conversations N times and report mean/std + bootstrap CI.

Legacy form ``python -m agents conv.json --agent react`` is mapped to ``annotate``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AnnotationResult
from .data_validation import strip_annotations, validate_file
from .evaluation import evaluate_dataset, headline, summarize_runs


def _load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(obj: Any, path: Optional[str]) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)


def _expand(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return out


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent", choices=["react", "multi_agent"], default="react")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None, help="Sampling seed (OpenAI only; best effort)")
    p.add_argument("--json-mode", action="store_true", help="Request JSON-object responses")
    p.add_argument("--no-critic", action="store_true", help="multi_agent: skip the critic")
    p.add_argument("--coordinator", action="store_true", help="multi_agent: use the coordinator")
    p.add_argument("--no-relations", action="store_true")


def _make_runner(a: argparse.Namespace):
    from .runner import AnnotationRunner

    return AnnotationRunner(
        agent_type=a.agent, provider=a.provider, model=a.model,
        temperature=a.temperature, seed=a.seed, json_mode=a.json_mode,
        use_critic=not a.no_critic, use_coordinator=a.coordinator,
    )


def cmd_validate(a: argparse.Namespace) -> int:
    bad = 0
    for path in _expand(a.paths):
        issues = validate_file(path)
        print(f"{'OK  ' if not issues else 'FAIL'} {path}")
        for i in issues:
            print(f"     - {i}")
        bad += bool(issues)
    return 1 if bad else 0


def cmd_annotate(a: argparse.Namespace) -> int:
    conv = _load(Path(a.input))
    if a.strip_gold:
        conv = strip_annotations(conv)
    runner = _make_runner(a)
    result = runner.annotate(conv, include_relations=not a.no_relations)
    runner.save_result(result, a.output) if a.output else print(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    diag = result.metadata.get("diagnostics", {})
    if diag:
        print(f"diagnostics: {json.dumps(diag)}", file=sys.stderr)
    return 0


def _pairs(pred_paths: List[Path], gold_paths: List[Path]):
    gold = {str(_load(p).get("id", p.stem)): _load(p) for p in gold_paths}
    pairs, missing = [], []
    for p in pred_paths:
        d = _load(p)
        cid = str(d.get("conversation_id", d.get("id", p.stem)))
        if cid in gold:
            pairs.append((AnnotationResult.from_dict(d), gold[cid]))
        else:
            missing.append(str(p))
    return pairs, missing


def cmd_evaluate(a: argparse.Namespace) -> int:
    pairs, missing = _pairs(_expand(a.pred), _expand(a.gold))
    if not pairs:
        print("no prediction matched a gold conversation id", file=sys.stderr)
        return 2
    report = evaluate_dataset(pairs, n_boot=a.n_boot, seed=a.boot_seed)
    if missing:
        report["unmatched_predictions"] = missing
    _write_json(report, a.output)
    return 0


def cmd_experiment(a: argparse.Namespace) -> int:
    gold_paths = _expand(a.gold)
    gold = [_load(p) for p in gold_paths]
    for p, g in zip(gold_paths, gold):
        problems = validate_file(p)
        if problems:
            print(f"refusing to run: {p} is invalid: {problems[:3]}", file=sys.stderr)
            return 2
    runner = _make_runner(a)
    runs, reports = [], []
    for k in range(a.runs):
        pairs = [(runner.annotate(strip_annotations(g), include_relations=not a.no_relations), g)
                 for g in gold]
        rep = evaluate_dataset(pairs, n_boot=a.n_boot, seed=a.boot_seed + k)
        rep["usage"] = runner.agent.usage() if hasattr(runner.agent, "usage") else {}
        reports.append(rep)
        runs.append(rep["headline"])
        print(f"run {k + 1}/{a.runs}: {json.dumps(rep['headline'])}", file=sys.stderr)
    out = {"config": runner.config.to_dict() | {"api_key": None}, "n_runs": a.runs,
           "aggregate": summarize_runs(runs), "runs": reports}
    _write_json(out, a.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m agents", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate conversation JSON files")
    v.add_argument("paths", nargs="+")
    v.set_defaults(func=cmd_validate)

    an = sub.add_parser("annotate", help="annotate a conversation")
    an.add_argument("input")
    an.add_argument("-o", "--output")
    an.add_argument("--strip-gold", action="store_true", help="drop existing annotations first")
    _add_agent_args(an)
    an.set_defaults(func=cmd_annotate)

    ev = sub.add_parser("evaluate", help="score predictions against gold")
    ev.add_argument("--pred", nargs="+", required=True)
    ev.add_argument("--gold", nargs="+", required=True)
    ev.add_argument("--n-boot", type=int, default=1000)
    ev.add_argument("--boot-seed", type=int, default=0)
    ev.add_argument("-o", "--output")
    ev.set_defaults(func=cmd_evaluate)

    ex = sub.add_parser("experiment", help="repeated annotate+evaluate on gold data")
    ex.add_argument("--gold", nargs="+", required=True)
    ex.add_argument("--runs", type=int, default=3)
    ex.add_argument("--n-boot", type=int, default=1000)
    ex.add_argument("--boot-seed", type=int, default=0)
    ex.add_argument("-o", "--output")
    _add_agent_args(ex)
    ex.set_defaults(func=cmd_experiment)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].endswith(".json"):  # legacy: python -m agents conv.json ...
        argv = ["annotate"] + argv
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
