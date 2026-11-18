"""Run a verifier over another configuration's stored answers.

For each task, the verifier works the task as any run does (same tables, tools and step budget), but every turn's
query is followed by the values the answering run stored for that turn's outputs, with the instruction to check each
against the data by recomputing it and to store the value it finds correct. Only the values are shown, not the
answering run's code or prose. The verifier's stored outputs are scored inline by the task's validators, as in any
run; the scorer is given the same extended query, so it does not refuse the turn as edited.

Studies are kept under experiments_verify/, apart from experiments/, so the paper's extraction never reads them.

    uv run python -m scripts.run_verifier --answers experiments/deepseek-v4-pro-0813-thinking-20261006 \\
        --answer-run 1 --model gpt-6-luna-thinking --tasks analysis/verification_20261116/tasks.json \\
        --experiment deepseek-v4-pro-0813-thinking__gpt-6-luna-thinking-20261116 --concurrency 5
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from config import BENCHMARKS_JSON, MODELS_TOML
from core.evaluator import load_specs
from core.models import load_tool_gateway
from core.results import new_run_id
from scripts.run import _check_harness, _pending_specs, run_cases

VERIFY_DIR = Path("experiments_verify")
INSTRUCTION = ("Another analyst stored these values for this turn's outputs. Check each against the data by "
               "recomputing it, and store the value you find correct for each output.")


def answer_run(study: Path, task: str, index: int) -> dict:
    """The run.json of the answering study's index-th run (1-based, in run-id order) of a task."""
    runs = sorted((study / "runs" / task).glob("*/*/run.json"), key=lambda p: p.parent.name)
    if len(runs) < index:
        raise FileNotFoundError(f"{study.name}/{task}: {len(runs)} runs, run {index} requested")
    return json.loads(runs[index - 1].read_text())


def stored_block(stores: list[str], outputs: dict) -> str:
    lines = [f"- {name} = {json.dumps(outputs[name], default=str)}" if outputs.get(name) is not None
             else f"- {name}: not stored" for name in stores]
    return INSTRUCTION + "\n" + "\n".join(lines)


def verifier_spec(spec: dict, answered: dict) -> dict:
    """The task with each turn's query followed by the answering run's stored values for that turn."""
    out = copy.deepcopy(spec)
    stored = {c["id"]: c["turns"] for c in answered["conversations"]}
    for conversation in out["conversations"]:
        cid = conversation.get("id", "main")   # the default id core.types.Conversation gives an unnamed conversation
        turns = stored[cid]
        if len(turns) != len(conversation["turns"]):
            raise ValueError(f"{spec['name']}/{cid}: answering run has {len(turns)} turns")
        for turn, answer in zip(conversation["turns"], turns):
            turn["query"] = turn["query"] + "\n\n" + stored_block(turn["stores"], answer.get("outputs") or {})
    out["verification"] = {"answering_study": answered["run"]["study_id"], "answering_run": answered["run"]["run_id"]}
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, type=Path, help="Answering study directory under experiments/.")
    parser.add_argument("--answer-run", type=int, default=1, help="Which of its runs, 1-based in run-id order.")
    parser.add_argument("--model", required=True, help="Verifier section in models.toml.")
    parser.add_argument("--tasks", required=True, type=Path, help="JSON with the task lists to verify.")
    parser.add_argument("--case", nargs="*", dest="case_names", help="Restrict to these tasks (smoke runs).")
    parser.add_argument("--experiment", required=True, help="Study id under experiments_verify/.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--step-budget", type=int, default=30)
    args = parser.parse_args()

    selected = json.loads(args.tasks.read_text())
    names = set(selected["shared"]) | set(selected["other"])
    if args.case_names:
        names &= set(args.case_names)
    specs = [verifier_spec(spec, answer_run(args.answers, spec["name"], args.answer_run))
             for spec in load_specs(BENCHMARKS_JSON) if spec["name"] in names]
    model, cfg = load_tool_gateway(MODELS_TOML, args.model)
    experiment_dir = (VERIFY_DIR / args.experiment).resolve()
    _check_harness(experiment_dir, "tool")
    specs, skipped = _pending_specs(experiment_dir, cfg, specs)
    print(f"SKIP completed={skipped} pending={len(specs)}", flush=True)
    if not specs:
        print(f"EXPERIMENT {experiment_dir}\nNOOP all requested tasks already complete")
        return
    run_id = new_run_id()
    print(f"EXPERIMENT {experiment_dir} RUN {run_id} MODEL {cfg.api_model} ANSWERS {args.answers.name} run {args.answer_run}", flush=True)
    summary = await run_cases(model, cfg, specs, experiment_dir, concurrency=args.concurrency, run_id=run_id,
                              arm="verify", score_inline=True, total_step_budget=args.step_budget,
                              injection="lazy", harness="tool")
    print(f"SUMMARY {summary['passed']}/{summary['total']} passed completed={summary['completed']} "
          f"infrastructure_errors={summary['infrastructure_errors']} unanswered={summary['unanswered']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
