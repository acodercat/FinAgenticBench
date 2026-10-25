"""Batch Response Quality scoring with sample-level resume semantics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
from typing import Callable

from config import BENCHMARKS_JSON, MODELS_TOML, PROJECT_ROOT
from core.evaluator import load_specs
from core.judging import (
    DEFAULT_JUDGE_PANEL, JudgeCallTracker, call_json,
    evaluate_panel, infer_tested_lab, load_channel, rater_pool,
    reconcile_prior_block, write_channel,
)
from core.models import ModelConfig, ModelRegistry
from core.results import (
    aggregates_dir, atomic_write_json, channel_path, iter_result_files, panel_label,
    load_result,
    programmatic_verification, resolve_experiment, utc_now,
)
from core.scoring import Criteria, aggregate_quality, validate_quality_judge
from core.types import conversations_of, sample_id


CRITERIA_PATH = PROJECT_ROOT / "criteria.json"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "judge_prompt.md"
SCHEMA = "finbench-response-quality-v2"


def criteria_text(criteria: Criteria) -> str:
    lines = []
    for dimension in criteria.dimensions:
        lines.append(
            f"### {dimension.id}: {dimension.name} (weight {dimension.weight})\n"
            f"{dimension.description}"
        )
        lines.extend(f"- {level}: {dimension.anchors[level]}" for level in range(1, 6))
    return "\n".join(lines)


def reference_outputs(spec: dict) -> dict:
    frozen = spec.get("reference_outputs")
    if isinstance(frozen, dict):
        return frozen
    module = importlib.import_module(spec["module"])
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def conversation_context(history: list[tuple[str, str]]) -> str:
    """Render the turns before this one, as context and never as the subject.

    A later turn refers back — "those filings", "that same period" — so a judge
    shown only the final turn cannot resolve the scope it is grading. The
    templates state the rule that keeps the channel honest: score only the final
    turn.
    """
    if not history:
        return "(This is the first turn of the conversation.)"
    blocks = []
    for index, (query, response) in enumerate(history, start=1):
        blocks.append(
            f"### Earlier turn {index} — question\n\n````text\n{query}\n````\n\n"
            f"### Earlier turn {index} — agent's response\n\n````text\n{response}\n````"
        )
    return "\n\n".join(blocks)


def prompt_renderer(
    spec: dict, result: dict, criteria: Criteria, context: str = ""
) -> Callable[[str], str]:
    """Return a per-judge renderer, resolving the reference outputs only once.

    Only the judge's own name varies between panel members, and recomputing
    ground truth per judge would run each case's validator three times.
    """
    template = PROMPT_PATH.read_text()
    fields = {
        "criteria": criteria_text(criteria),
        "query": spec["query"],
        "reference_outputs": json.dumps(
            reference_outputs(spec), default=str, indent=2
        ),
        "judge_reference": json.dumps(
            spec.get("judge_reference", {}), ensure_ascii=False, indent=2
        ),
        "agent_outputs": json.dumps(result.get("outputs", {}), default=str, indent=2),
        "response": result.get("response", ""),
        "conversation_context": context or conversation_context([]),
    }
    return lambda judge_id: template.format(judge_id=judge_id, **fields)


def _block(
    criteria: Criteria,
    criteria_path: Path,
    judges: list[ModelConfig],
    exclude_same_lab: bool,
    minimum_judges: int,
) -> dict:
    """The settings a stored sample was produced under; see reconcile_prior_block.

    `criteria_path` is recorded for provenance only. What actually invalidates a
    resume is `criteria_fingerprint`, which covers the rubric's content, so an
    A/B file with identical anchors correctly resumes and an edited one does not.
    """
    return {
        "criteria_fingerprint": criteria.fingerprint(),
        "criteria_path": str(criteria_path),
        "requested_panel": [judge.public_fingerprint() for judge in judges],
        "exclude_same_lab": exclude_same_lab,
        "params": {
            "minimum_successful_judges": minimum_judges,
        },
    }


def score_result_file(
    result_path: Path,
    current_spec: dict,
    registry: ModelRegistry,
    judge_names: list[str],
    tracker: JudgeCallTracker,
    *,
    exclude_same_lab: bool = False,
    minimum_judges: int = 2,
    rescore: bool = False,
    judge_call: Callable[[ModelConfig, str], dict] = call_json,
    criteria_path: Path = CRITERIA_PATH,
) -> str:
    saved = load_result(result_path)
    sibling = channel_path(result_path, "response_quality", panel_label(judge_names))
    criteria = Criteria.from_json(criteria_path)
    requested = registry.select(judge_names)
    block = _block(
        criteria, criteria_path, requested, exclude_same_lab, minimum_judges,
    )
    entries, prior_block = load_channel(sibling, SCHEMA)
    discarded_prior = False
    if entries and not reconcile_prior_block(
        sibling, prior_block, block, rescore=rescore
    ):
        entries, discarded_prior = {}, True

    spec = current_spec
    turns = {c.id: c.turns for c in conversations_of(spec)}
    tested_lab = infer_tested_lab(saved, registry)
    judges = rater_pool(tested_lab, requested, exclude_same_lab=exclude_same_lab)
    pv_success, _ = programmatic_verification(result_path)

    layout, outcomes = [], []
    for conversation in saved["conversations"]:
        identifiers, history = [], []
        for index, stored in enumerate(conversation["turns"]):
            identifier = sample_id(current_spec["name"], conversation["id"], index)
            identifiers.append(identifier)
            spec_turn = turns[conversation["id"]][index]
            context = conversation_context(history)
            history.append((stored.get("query", ""), stored.get("response", "")))
            if entries.get(identifier, {}).get("status") == "scored" and not rescore:
                outcomes.append("already_scored")
                continue
            if not str(stored.get("response", "")).strip():
                entries[identifier] = {"status": "skipped", "reason": "empty_response"}
                outcomes.append("skipped")
                continue
            render = prompt_renderer(
                {**spec, "query": spec_turn.query,
                 "judge_reference": spec.get("judge_reference", {})},
                stored, criteria, context,
            )
            panel = evaluate_panel(
                judges,
                # Bound now, not captured: the panel runs inside this
                # iteration, but a late-bound `render` would be a live bug the
                # moment this call is deferred.
                lambda judge, render=render: validate_quality_judge(
                    judge_call(judge, render(judge.name)), criteria
                ),
                tracker=tracker, minimum=minimum_judges,
            )
            skip_reason = panel.skip_reason()
            if skip_reason:
                entries[identifier] = {"status": "skipped", "reason": skip_reason}
                outcomes.append("skipped")
                continue
            scores = aggregate_quality(
                [value for _, value in panel.successes], criteria
            )
            if not panel.complete:
                scores["flag_reasons"] = scores["flag_reasons"] + [
                    f"partial_panel:{panel.responded}/{panel.requested}_judges_responded"
                ]
                scores["flagged_for_review"] = True
            entries[identifier] = {
                "status": "scored",
                "tested_model": saved["model"],
                "programmatic_verification_passed": pv_success,
                "response_quality": scores,
                "effective_panel": [
                    judge.public_fingerprint() for judge, _ in panel.successes
                ],
                "judges_expected": panel.requested,
                "partial_panel": not panel.complete,
                "judge_failures": panel.failures,
            }
            outcomes.append("scored")
        layout.append({"id": conversation["id"], "sample_ids": identifiers})

    write_channel(
        sibling, schema=SCHEMA, case=current_spec["name"], conversations=layout,
        block={**block, "updated_at": utc_now(), "judge_health": tracker.summary()},
        entries=entries,
    )
    if outcomes and all(item == "already_scored" for item in outcomes):
        return "already_scored"
    if "scored" in outcomes:
        return "rescored" if discarded_prior else "scored"
    return "skipped"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--case", nargs="*", dest="case_names")
    parser.add_argument("--judge", nargs="+", default=list(DEFAULT_JUDGE_PANEL))
    parser.add_argument(
        "--minimum-judges", type=int, default=2,
        help="Minimum successful judges; use 1 only for an explicitly exploratory arm.",
    )
    parser.add_argument("--exclude-same-lab", action="store_true")
    parser.add_argument(
        "--criteria", type=Path, default=CRITERIA_PATH, metavar="PATH",
        help="Score against an alternative rubric, for calibration or an A/B "
             "sensitivity run, without editing the frozen criteria.json.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--rescore", "--force", dest="rescore", action="store_true",
        help="Recompute samples stored under different scoring settings.",
    )
    args = parser.parse_args()

    if len(args.judge) != len(set(args.judge)):
        raise ValueError("judge names must be unique")
    if not 1 <= args.minimum_judges <= len(args.judge):
        raise ValueError("minimum judges must be between 1 and the requested panel size")
    if not 1 <= args.concurrency <= 16:
        raise ValueError("concurrency must be between 1 and 16")
    experiment = resolve_experiment(args.experiment)
    specs = {spec["name"]: spec for spec in load_specs(BENCHMARKS_JSON)}
    requested_cases = set(args.case_names) if args.case_names else None
    registry = ModelRegistry.load(MODELS_TOML)
    registry.select(args.judge)   # fail before any call on an unknown judge name
    tracker = JudgeCallTracker()
    counts: dict[str, int] = {}
    jobs = list(iter_result_files(experiment, requested_cases))
    for case_name, _ in jobs:
        if case_name not in specs:
            raise ValueError(f"result directory names unknown case {case_name!r}")

    def score(job: tuple[str, Path]) -> tuple[str, str]:
        case_name, path = job
        outcome = score_result_file(
            path, specs[case_name], registry, args.judge,
            tracker, exclude_same_lab=args.exclude_same_lab,
            rescore=args.rescore,
            minimum_judges=args.minimum_judges,
            criteria_path=args.criteria,
        )
        return case_name, outcome

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outcomes = pool.map(score, jobs)
        for case_name, outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
            print(f"{case_name}: {outcome}", flush=True)

    panel = panel_label(args.judge)
    atomic_write_json(aggregates_dir(experiment, panel) / "response_quality_judge_health.json", {
        "schema": "finbench-judge-health-v1",
        "channel": "response_quality",
        "updated_at": utc_now(),
        "judges": tracker.summary(),
    })
    from scripts.run_stats import write_reports
    write_reports(experiment, panel)
    print(f"SUMMARY {counts} judge_health={tracker.summary()}")


if __name__ == "__main__":
    main()
