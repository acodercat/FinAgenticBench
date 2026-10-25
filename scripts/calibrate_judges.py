"""Create human-rating templates and compare frozen LLM judges with humans."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import statistics
from pathlib import Path
from typing import Any

from config import BENCHMARKS_JSON, PROJECT_ROOT
from core.evaluator import load_specs
from core.results import (
    aggregates_dir, atomic_write_json, atomic_write_text, channel_path,
    experiment_panels, iter_result_files, resolve_experiment,
    result_identity, utc_now,
)
from core.scoring import Criteria, claim_mode, krippendorff_alpha
from core.types import conversations_of, sample_id


SCHEMA = "finbench-human-judge-calibration-v1"
RQ_SCHEMA = "finbench-response-quality-v2"
CC_SCHEMA = "finbench-claim-coverage-v2"


def rating_id(case: str, model: str, run_id: str, turn: str) -> str:
    """Address one rated turn. Distinct from `core.types.sample_id`, which
    addresses a turn within a case; a calibration row also names the run it
    came from, because two runs of the same turn are two things to rate."""
    return f"{case}/{model}/{run_id}/{turn.rsplit('/', 2)[-2]}/{turn.rsplit('/', 1)[-1]}"


def _channel_turns(path: Path, schema: str) -> dict[str, dict]:
    """Every scored turn in a run-shaped channel file, by its sample id."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported channel schema")
    return {
        turn["sample_id"]: turn
        for conversation in payload.get("conversations", [])
        for turn in conversation.get("turns", [])
        if turn.get("status") == "scored" and turn.get("sample_id")
    }


def load_llm_samples(experiment: Path, panel: str) -> dict[str, dict[str, Any]]:
    specs = {spec["name"]: spec for spec in load_specs(BENCHMARKS_JSON)}
    output = {}
    for case, path in iter_result_files(experiment):
        _, model, run_id = result_identity(path)
        snapshot = specs[case]
        rq_turns = _channel_turns(channel_path(path, "response_quality", panel), RQ_SCHEMA)
        cc_turns = _channel_turns(channel_path(path, "claim_coverage", panel), CC_SCHEMA)
        for conversation in conversations_of(snapshot):
            for index, turn in enumerate(conversation.turns):
                key = sample_id(case, conversation.id, index)
                rq = rq_turns.get(key, {}).get("response_quality", {})
                cc = cc_turns.get(key, {}).get("claim_coverage", {})
                output[rating_id(case, model, run_id, key)] = {
                    "case": case, "model": model, "run_id": run_id,
                    "conversation": conversation.id, "turn": index + 1,
                    "claims": turn.claims,
                    "rq_judges": rq.get("judges", []),
                    "cc_judges": cc.get("judges", []),
                }
    return output


def write_template(experiment: Path, output: Path, panel: str, *, force: bool = False) -> int:
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    rows = []
    for identifier, sample in sorted(load_llm_samples(experiment, panel).items()):
        rows.append({
            "sample_id": identifier, "case": sample["case"], "model": sample["model"],
            "run_id": sample["run_id"], "rater_id": "REPLACE_WITH_BLINDED_RATER_ID",
            "rq_scores": {dimension.id: None for dimension in criteria.dimensions},
            "claim_coverage": {claim["id"]: None for claim in sample["claims"]},
        })
    atomic_write_text(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return len(rows)


def load_human_ratings(path: Path, criteria: Criteria, llm_samples: dict[str, dict]) -> list[dict]:
    dimensions = {dimension.id for dimension in criteria.dimensions}
    ratings, identities = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        identifier, rater = row.get("sample_id"), row.get("rater_id")
        if identifier not in llm_samples:
            raise ValueError(f"{path}:{line_number}: unknown sample_id {identifier!r}")
        if not isinstance(rater, str) or not rater.strip() or rater.startswith("REPLACE_"):
            raise ValueError(f"{path}:{line_number}: a blinded rater_id is required")
        if (identifier, rater) in identities:
            raise ValueError(f"{path}:{line_number}: duplicate sample/rater rating")
        scores = row.get("rq_scores", {})
        if set(scores) != dimensions or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5
            for value in scores.values()
        ):
            raise ValueError(f"{path}:{line_number}: rq_scores must contain every 1-5 dimension")
        claims = row.get("claim_coverage", {})
        expected_claims = {claim["id"] for claim in llm_samples[identifier]["claims"]}
        if set(claims) != expected_claims or any(not isinstance(value, bool) for value in claims.values()):
            raise ValueError(f"{path}:{line_number}: claim_coverage must contain every boolean claim")
        identities.add((identifier, rater))
        ratings.append({
            "sample_id": identifier, "rater_id": rater,
            "rq_scores": scores, "claim_coverage": claims,
        })
    if not ratings:
        raise ValueError("human ratings file contains no completed ratings")
    return ratings


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def build_calibration_report(
    experiment: Path, ratings_path: Path, panel: str,
) -> dict[str, Any]:
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    llm_samples = load_llm_samples(experiment, panel)
    ratings = load_human_ratings(ratings_path, criteria, llm_samples)
    human_by_sample: dict[str, list[dict]] = defaultdict(list)
    for rating in ratings:
        human_by_sample[rating["sample_id"]].append(rating)
    under_rated = sorted(
        identifier for identifier, values in human_by_sample.items()
        if len({value["rater_id"] for value in values}) < 2
    )
    missing_llm_rq = sorted(
        identifier for identifier in human_by_sample
        if len(llm_samples[identifier]["rq_judges"]) < 2
    )
    missing_llm_cc = sorted(
        identifier for identifier in human_by_sample
        if llm_samples[identifier]["claims"]
        and len(llm_samples[identifier]["cc_judges"]) < 2
    )

    dimensions = {}
    for dimension in criteria.dimensions:
        human_units, llm_units, human_means, llm_means = [], [], [], []
        for identifier, human in sorted(human_by_sample.items()):
            llm = llm_samples[identifier]["rq_judges"]
            human_values = [float(row["rq_scores"][dimension.id]) for row in human]
            llm_values = [float(row["scores"][dimension.id]) for row in llm if dimension.id in row.get("scores", {})]
            if len(human_values) >= 2:
                human_units.append(human_values)
            if len(llm_values) >= 2:
                llm_units.append(llm_values)
            if human_values and llm_values:
                human_means.append(statistics.fmean(human_values))
                llm_means.append(statistics.fmean(llm_values))
        dimensions[dimension.id] = {
            "compared_samples": len(human_means),
            "human_inter_rater_alpha_interval": krippendorff_alpha(human_units, "interval"),
            "llm_inter_judge_alpha_interval": krippendorff_alpha(llm_units, "interval"),
            "human_llm_mean_absolute_error": _mean_abs_error(human_means, llm_means),
            "human_llm_pearson": _pearson(human_means, llm_means),
            "human_mean": statistics.fmean(human_means) if human_means else None,
            "llm_mean": statistics.fmean(llm_means) if llm_means else None,
        }

    human_claim_units, llm_claim_units = [], []
    agreement_values, by_mode = [], defaultdict(list)
    ties = 0
    for identifier, human in sorted(human_by_sample.items()):
        sample = llm_samples[identifier]
        llm_judges = sample["cc_judges"]
        modes = {claim["id"]: claim_mode(claim) for claim in sample["claims"]}
        for claim_id, mode in modes.items():
            human_values = [int(row["claim_coverage"][claim_id]) for row in human]
            llm_values = [int(row["coverage"][claim_id]) for row in llm_judges if claim_id in row.get("coverage", {})]
            if len(human_values) >= 2:
                human_claim_units.append(human_values)
            if len(llm_values) >= 2:
                llm_claim_units.append(llm_values)
            if human_values and llm_values:
                human_rate, llm_rate = statistics.fmean(human_values), statistics.fmean(llm_values)
                if human_rate == 0.5 or llm_rate == 0.5:
                    ties += 1
                    continue
                value = int((human_rate > 0.5) == (llm_rate > 0.5))
                agreement_values.append(value)
                by_mode[mode].append(value)
    compared_samples = len(human_by_sample)
    report = {
        "schema": SCHEMA, "generated_at": utc_now(), "experiment": experiment.name,
        "ratings_file": str(ratings_path),
        "coverage": {
            "available_experiment_samples": len(llm_samples),
            "human_rated_samples": compared_samples,
            "human_raters": sorted({rating["rater_id"] for rating in ratings}),
            "ratings": len(ratings), "samples_with_fewer_than_two_raters": under_rated,
            "rated_samples_with_fewer_than_two_llm_rq_judges": missing_llm_rq,
            "rated_samples_with_fewer_than_two_llm_cc_judges": missing_llm_cc,
        },
        "response_quality": {"dimensions": dimensions},
        "claim_coverage": {
            "human_inter_rater_alpha_nominal": krippendorff_alpha(human_claim_units, "nominal"),
            "llm_inter_judge_alpha_nominal": krippendorff_alpha(llm_claim_units, "nominal"),
            "majority_agreement": statistics.fmean(agreement_values) if agreement_values else None,
            "compared_claim_instances_excluding_ties": len(agreement_values),
            "excluded_majority_ties": ties,
            "majority_agreement_by_mode": {
                mode: statistics.fmean(values) for mode, values in sorted(by_mode.items()) if values
            },
        },
        "publication_readiness": {
            "coverage_ready": (
                compared_samples >= 30 and not under_rated
                and not missing_llm_rq and not missing_llm_cc
            ),
            "recommended_sample_range": "30-50 varied answers",
            "empirical_acceptance_thresholds": "must be preregistered by researchers; not invented by this tool",
            "ready": False,
            "note": (
                "Coverage readiness is necessary but not sufficient. Researchers must review disagreement, "
                "set acceptance thresholds before model ranking, and explicitly approve calibration."
            ),
        },
    }
    return report


def _mean_abs_error(left: list[float], right: list[float]) -> float | None:
    return statistics.fmean(abs(a - b) for a, b in zip(left, right)) if left else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--judge-panel", help="the judge panel to calibrate; required when the study has several")
    args = parser.parse_args()
    if bool(args.write_template) == bool(args.ratings):
        raise ValueError("choose exactly one of --write-template or --ratings")
    experiment = resolve_experiment(args.experiment)
    panel = args.judge_panel
    if panel is None:
        found = experiment_panels(experiment)
        if len(found) != 1:
            raise ValueError(f"choose a judge panel with --judge-panel; this study has {found or 'none'}")
        panel = found[0]
    if args.write_template:
        count = write_template(experiment, args.write_template, panel, force=args.force)
        print(json.dumps({"template": str(args.write_template), "samples": count}, indent=2))
        return
    report = build_calibration_report(experiment, args.ratings, panel)
    output = aggregates_dir(experiment, panel) / "human_judge_calibration.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
