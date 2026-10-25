"""Batch Claim Coverage scoring with sample-level resume semantics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Callable

from config import BENCHMARKS_JSON, MODELS_TOML, PROJECT_ROOT
from core.evaluator import load_specs
from scripts.score_response_quality import conversation_context
from core.judging import (
    DEFAULT_JUDGE_PANEL, JudgeCallTracker, call_json,
    evaluate_panel, infer_tested_lab, load_channel, rater_pool,
    reconcile_prior_block, write_channel,
)
from core.models import ModelConfig, ModelRegistry
from core.results import (
    aggregates_dir, atomic_write_json, channel_path, iter_result_files, panel_label,
    load_result,
    resolve_experiment, utc_now,
)
from core.scoring import aggregate_claims, validate_claim_judge
from core.types import conversations_of, sample_id


PROMPT_PATH = PROJECT_ROOT / "prompts" / "claim_verify_prompt.md"
SCHEMA = "finbench-claim-coverage-v2"


def prompt_renderer(
    query: str, claims: list, result: dict, context: str = ""
) -> Callable[[str], str]:
    """Return a per-judge renderer; only the judge's own name varies."""
    template = PROMPT_PATH.read_text()
    fields = {
        "query": query,
        "claims": json.dumps(claims, ensure_ascii=False, indent=2),
        "response": result.get("response", ""),
        "conversation_context": context or conversation_context([]),
    }
    return lambda judge_id: template.format(judge_id=judge_id, **fields)


def _block(
    judges: list[ModelConfig],
    exclude_same_lab: bool,
    minimum_judges: int,
) -> dict:
    """The settings a stored sample was produced under; see reconcile_prior_block.

    No criteria fingerprint: coverage is a boolean per claim, with no rubric and
    no cap for a criteria change to invalidate.
    """
    return {
        "requested_panel": [judge.public_fingerprint() for judge in judges],
        "exclude_same_lab": exclude_same_lab,
        "params": {
            "minimum_successful_judges": minimum_judges,
        },
    }


def verify_result_file(
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
) -> str:
    saved = load_result(result_path)
    sibling = channel_path(result_path, "claim_coverage", panel_label(judge_names))
    requested = registry.select(judge_names)
    block = _block(
        requested, exclude_same_lab, minimum_judges,
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
            # Claims are declared per turn, so a turn without them is out of
            # scope for this channel rather than a failure to judge.
            if not spec_turn.claims:
                entries[identifier] = {"status": "no_claims"}
                outcomes.append("skipped")
                continue
            if not str(stored.get("response", "")).strip():
                entries[identifier] = {"status": "skipped", "reason": "empty_response"}
                outcomes.append("skipped")
                continue
            render = prompt_renderer(
                spec_turn.query, spec_turn.claims, stored, context
            )
            panel = evaluate_panel(
                judges,
                # Bound now, not captured; see score_responses for why.
                lambda judge, render=render, claims=spec_turn.claims:
                    validate_claim_judge(judge_call(judge, render(judge.name)), claims),
                tracker=tracker, minimum=minimum_judges,
            )
            skip_reason = panel.skip_reason()
            if skip_reason:
                entries[identifier] = {"status": "skipped", "reason": skip_reason}
                outcomes.append("skipped")
                continue
            coverage = aggregate_claims(
                [value for _, value in panel.successes], spec_turn.claims
            )
            if not panel.complete:
                coverage["flag_reasons"] = coverage["flag_reasons"] + [
                    f"partial_panel:{panel.responded}/{panel.requested}_judges_responded"
                ]
                coverage["flagged_for_review"] = True
            entries[identifier] = {
                "status": "scored",
                "tested_model": saved["model"],
                "claim_coverage": coverage,
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

    def verify(job: tuple[str, Path]) -> tuple[str, str]:
        case_name, path = job
        outcome = verify_result_file(
            path, specs[case_name], registry, args.judge,
            tracker, exclude_same_lab=args.exclude_same_lab,
            rescore=args.rescore,
            minimum_judges=args.minimum_judges,
        )
        return case_name, outcome

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outcomes = pool.map(verify, jobs)
        for case_name, outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
            print(f"{case_name}: {outcome}", flush=True)

    panel = panel_label(args.judge)
    atomic_write_json(aggregates_dir(experiment, panel) / "claim_coverage_judge_health.json", {
        "schema": "finbench-judge-health-v1",
        "channel": "claim_coverage",
        "updated_at": utc_now(),
        "judges": tracker.summary(),
    })
    from scripts.run_stats import write_reports
    write_reports(experiment, panel)
    print(f"SUMMARY {counts} judge_health={tracker.summary()}")


if __name__ == "__main__":
    main()
