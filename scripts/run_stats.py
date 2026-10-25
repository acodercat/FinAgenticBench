"""Build publication-oriented experiment score, reliability, and review reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from config import BENCHMARKS_JSON, PROJECT_ROOT
from core.evaluator import load_specs
from core.results import (
    RUNS_DIR, aggregates_dir, atomic_write_json, atomic_write_text, channel_path, experiment_panels,
    iter_result_files, load_result, programmatic_verification, resolve_experiment,
    result_identity, result_is_complete, utc_now,
)
from core.types import conversations_of, sample_id
from core.scoring import (
    COMPOSITE_WEIGHT_SCENARIOS, Criteria, claim_mode, composite_score,
    compute_corpus_agreement, pass_hat_k,
)


RUN_STATS_SCHEMA = "finbench-run-stats-v3"
AGREEMENT_SCHEMA = "finbench-corpus-agreement-v1"
RQ_SCHEMA = "finbench-response-quality-v2"
CC_SCHEMA = "finbench-claim-coverage-v2"
BOOTSTRAP_DRAWS = 2000


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16)


def _metric_summary(records: list[dict[str, Any]], field: str, *, label: str) -> dict:
    """Average repeats inside model×arm×case, then bootstrap those case units."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    trajectories = 0
    for record in records:
        value = record.get(field)
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)):
            continue
        trajectories += 1
        grouped[(record["model"], record["arm"], record["case"])].append(float(value))
    unit_values = [statistics.fmean(values) for values in grouped.values()]
    if not unit_values:
        interval = None
    elif len(unit_values) == 1:
        interval = [unit_values[0], unit_values[0]]
    else:
        # Cluster bootstrap: sample case units and then repeats within each unit.
        rng = random.Random(_stable_seed(label))
        clusters = list(grouped.values())
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            selected = [rng.choice(clusters) for _ in clusters]
            draws.append(statistics.fmean(
                statistics.fmean(rng.choice(cluster) for _ in cluster)
                for cluster in selected
            ))
        interval = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
    return {
        "n_trajectories": trajectories,
        "n_case_units": len(unit_values),
        "mean": _mean(unit_values),
        "ci95_hierarchical_bootstrap": interval,
        "bootstrap_draws": BOOTSTRAP_DRAWS if len(unit_values) > 1 else 0,
    }


def _sample(path: Path, schema: str, wanted: str) -> tuple[dict | None, str | None]:
    """Find one turn's entry in a run-shaped channel file.

    Every turn is written, so a turn the channel declined carries its own status
    and reason: "missing" and "declined" stay distinguishable.
    """
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, "invalid_json"
    if payload.get("schema") != schema:
        return None, "wrong_schema"
    for conversation in payload.get("conversations", []):
        for turn in conversation.get("turns", []):
            if turn.get("sample_id") != wanted:
                continue
            if turn.get("status") == "scored":
                return turn, None
            reason = turn.get("reason") or turn.get("status", "not_scored")
            return None, f"skipped: {reason}"
    return None, "missing_sample"


def _panel_names(panel: Any) -> list[str]:
    if not isinstance(panel, list):
        return []
    return [
        str(item["name"] if isinstance(item, dict) else item)
        for item in panel if (isinstance(item, str) or isinstance(item, dict) and item.get("name"))
    ]


def _panel_status(sample: dict | None) -> str:
    if not sample:
        return "missing"
    requested = _panel_names(sample.get("requested_panel"))
    effective = _panel_names(sample.get("effective_panel"))
    if not requested or not effective:
        return "unregistered_panel"
    if len(effective) == len(requested) and set(effective) == set(requested):
        return "full_primary"
    if effective and set(effective).issubset(set(requested)):
        return "partial_primary"
    return "other_panel"


def _judge_usage(
    sample: dict | None,
) -> list[dict[str, Any]]:
    aggregate = (
        sample.get("response_quality") or sample.get("claim_coverage") or {}
        if sample else {}
    )
    # A sample records the judges it asked for, with their published pricing, so
    # judge cost is read from the sample rather than from a separate declaration.
    pricing = {
        str(item["name"]): (
            item.get("input_cost_per_million_usd"),
            item.get("output_cost_per_million_usd"),
        )
        for item in ((sample or {}).get("requested_panel") or [])
        if isinstance(item, dict) and item.get("name")
    }
    rows = []
    for judge in aggregate.get("judges", []):
        usage = judge.get("_usage") if isinstance(judge, dict) else None
        if isinstance(usage, dict):
            row = {
                "judge": str(judge.get("judge_id", "unknown")),
                **{key: int(usage.get(key, 0) or 0) for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens"
                )},
            }
            input_price, output_price = pricing.get(row["judge"], (None, None))
            row["estimated_cost_usd"] = (
                row["prompt_tokens"] * input_price / 1_000_000
                + row["completion_tokens"] * output_price / 1_000_000
                if input_price is not None and output_price is not None else None
            )
            rows.append(row)
    return rows


def _usage_summary(records: list[dict]) -> dict:
    totals = Counter()
    by_channel: dict[str, Counter] = defaultdict(Counter)
    by_judge: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for channel in ("response_quality", "claim_coverage"):
            for usage in record.get("judge_usage", {}).get(channel, []):
                values = {key: int(usage.get(key, 0)) for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens"
                )}
                for counter in (totals, by_channel[channel], by_judge[usage["judge"]]):
                    counter.update(values)
                    counter["successful_parsed_calls"] += 1
                    if usage.get("estimated_cost_usd") is None:
                        counter["unpriced_successful_calls"] += 1
                    else:
                        counter["estimated_cost_usd"] += float(usage["estimated_cost_usd"])
    return {
        "scope": "successful judge calls with parsed usage; failed-call billing may be absent",
        **dict(totals),
        "by_channel": {key: dict(value) for key, value in sorted(by_channel.items())},
        "by_judge": {key: dict(value) for key, value in sorted(by_judge.items())},
    }


def _runtime_summary(records: list[dict]) -> dict:
    runtime = [record.get("runtime", {}) for record in records]
    def numbers(key: str) -> list[float]:
        return [float(item[key]) for item in runtime if isinstance(item.get(key), (int, float))]
    failures = Counter(str(item["pv_failure_type"]) for item in records if item.get("pv_failure_type"))
    priced_costs = numbers("estimated_cost_usd")
    return {
        "trajectories": len(records),
        "total_steps": int(sum(numbers("steps"))), "mean_steps": _mean(numbers("steps")),
        "prompt_tokens": int(sum(numbers("prompt_tokens"))),
        "completion_tokens": int(sum(numbers("completion_tokens"))),
        "total_tokens": int(sum(numbers("total_tokens"))),
        "mean_total_tokens": _mean(numbers("total_tokens")),
        "elapsed_seconds": sum(numbers("elapsed")),
        "mean_elapsed_seconds": _mean(numbers("elapsed")),
        "estimated_agent_cost_usd": sum(priced_costs) if priced_costs else None,
        "priced_trajectories": len(priced_costs),
        "unpriced_trajectories": len(records) - len(priced_costs),
        "failure_types": dict(sorted(failures.items())),
        "protocol_repair_attempted": sum(bool(item.get("protocol_repair_attempted")) for item in runtime),
        "protocol_repair_successes": sum(bool(item.get("protocol_repair_success")) for item in runtime),
    }


def _reliability(records: list[dict]) -> dict:
    """pass^k over the cases that were run at least k times.

    A mean pass rate treats one lucky trial as a pass; pass^k asks whether a
    case passes every time, so it separates a model that is right from one that
    is merely right sometimes. Cases run fewer than k times cannot answer that
    question and are excluded from that k rather than counted as failures.
    """
    trials: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for record in records:
        value = record.get("pv_passed")
        if isinstance(value, bool):
            trials[(record["model"], record["arm"], record["case"])].append(value)
    if not trials:
        return {"cases": 0, "trials_per_case": {}, "pass_hat_k": {}}
    counts = sorted(len(v) for v in trials.values())
    summary = {}
    for k in range(1, max(counts) + 1):
        eligible = [v for v in trials.values() if len(v) >= k]
        if not eligible:
            continue
        summary[str(k)] = {
            "value": _mean([pass_hat_k(len(v), sum(v), k) for v in eligible]),
            "cases": len(eligible),
        }
    return {
        "cases": len(trials),
        "trials_per_case": dict(sorted(Counter(counts).items())),
        "pass_hat_k": summary,
        "note": "pass^k is the share of k-trial subsets containing only passes, averaged over cases",
    }


def _rollup(records: list[dict], label: str) -> dict:
    dimension_ids = sorted({
        key.removeprefix("rq_dimension_all:")
        for record in records for key in record
        if key.startswith("rq_dimension_all:")
    })
    headline = {
        "response_quality": _metric_summary(records, "rq_headline", label=f"{label}:rq:primary"),
        "response_quality_by_dimension": {
            dimension: _metric_summary(
                records, f"rq_dimension_headline:{dimension}",
                label=f"{label}:rq:{dimension}:primary",
            ) for dimension in dimension_ids
        },
        "required_claim_coverage": _metric_summary(records, "cc_headline", label=f"{label}:cc:primary"),
        "conditional_violation_rate": _metric_summary(
            records, "conditional_violation_rate_headline", label=f"{label}:conditional:primary"
        ),
        "composite": {
            scenario: _metric_summary(records, f"composite_headline:{scenario}", label=f"{label}:composite:{scenario}:primary")
            for scenario in COMPOSITE_WEIGHT_SCENARIOS
        },
    }
    all_scored = {
        "response_quality": _metric_summary(records, "rq_all", label=f"{label}:rq:all"),
        "response_quality_by_dimension": {
            dimension: _metric_summary(
                records, f"rq_dimension_all:{dimension}",
                label=f"{label}:rq:{dimension}:all",
            ) for dimension in dimension_ids
        },
        "required_claim_coverage": _metric_summary(records, "cc_all", label=f"{label}:cc:all"),
        "conditional_violation_rate": _metric_summary(
            records, "conditional_violation_rate_all", label=f"{label}:conditional:all"
        ),
        "composite": {
            scenario: _metric_summary(records, f"composite_all:{scenario}", label=f"{label}:composite:{scenario}:all")
            for scenario in COMPOSITE_WEIGHT_SCENARIOS
        },
    }
    result = {
        "trajectories": len(records),
        "case_units": len({(r["model"], r["arm"], r["case"]) for r in records}),
        "programmatic_verification": _metric_summary(records, "pv_numeric", label=f"{label}:pv"),
        "reliability": _reliability(records),
        "variable_accuracy": _metric_summary(records, "pv_variable_accuracy", label=f"{label}:variable"),
        "headline_full_primary_panel": headline,
        "all_scored_panel_sensitivity": all_scored,
        "panel_status": {
            "response_quality": dict(sorted(Counter(r.get("rq_panel_status", "missing") for r in records).items())),
            "claim_coverage": dict(sorted(Counter(r.get("cc_panel_status", "missing") for r in records).items())),
        },
        "runtime": _runtime_summary(records), "judge_usage": _usage_summary(records),
    }
    result.update({
        "pv_pass_rate": result["programmatic_verification"]["mean"],
        "rq_mean_full_primary": headline["response_quality"]["mean"],
        "required_claim_coverage_full_primary": headline["required_claim_coverage"]["mean"],
    })
    return result


def _review_reasons(record: dict, rq: dict | None, cc: dict | None) -> list[str]:
    """Collect every reason this sample needs a human read.

    The per-sample judge reasons are taken from each aggregate's own
    `flag_reasons` rather than re-derived here: the thresholds live once, in
    `core.scoring`, beside the code that applied them, so the queue cannot
    disagree with the flag it is reporting. Everything else below is a reason
    only this report can see, because it compares channels, panels and
    expectations across the whole run.
    """
    reasons = []
    for key, label in (("pv_passed", "programmatic_verification"), ("rq_all", "response_quality"), ("cc_all", "claim_coverage")):
        if record.get(key) is None:
            reasons.append(f"missing_channel:{label}")
    for sample in (rq, cc):
        if sample and sample.get("flagged_for_review"):
            reasons.extend(sample.get("flag_reasons", []))
    for channel in ("rq", "cc"):
        status = record.get(f"{channel}_panel_status")
        if status not in {"full_primary", "missing"}:
            reasons.append(f"{channel}_panel:{status}")
    if record.get("rq_panel_status") != "missing" and record.get("cc_panel_status") != "missing" \
            and set(record.get("rq_effective_panel", [])) != set(record.get("cc_effective_panel", [])):
        reasons.append("panel_mismatch:rq_vs_cc")
    # A multi-source case answered correctly without the agent's code ever
    # naming one of its tables is either a shortcut the case did not intend or a
    # source that carries no analytic weight. Restricted to passing runs: a
    # failed run has its own reasons and would only dilute the queue.
    if (
        record.get("source_mode") == "multi"
        and record.get("pv_passed") is True
        and record.get("tables_unreferenced")
    ):
        reasons.append(
            "unreferenced_tables:" + ",".join(sorted(record["tables_unreferenced"]))
        )
    if record.get("pv_passed") is False and (record.get("rq_all") or 0) >= 4:
        reasons.append("pv_rq_discordance:pv_fail_high_rq")
    if record.get("pv_passed") is True and isinstance(record.get("rq_all"), (int, float)) and record["rq_all"] <= 2:
        reasons.append("pv_rq_discordance:pv_pass_low_rq")
    return sorted(set(reasons))


def _paired_differences(records: list[dict]) -> list[dict]:
    output = []
    fields = {
        "pv_pass_rate": "pv_numeric", "rq_full_primary": "rq_headline",
        "required_claim_coverage_full_primary": "cc_headline",
        "composite_primary_60_25_15_full_primary": "composite_headline:primary_60_25_15",
    }
    for arm in sorted({r["arm"] for r in records}):
        arm_records = [r for r in records if r["arm"] == arm]
        for model_a, model_b in combinations(sorted({r["model"] for r in arm_records}), 2):
            comparison = {"arm": arm, "model_a": model_a, "model_b": model_b,
                          "direction": "model_b_minus_model_a", "metrics": {}}
            for label, field in fields.items():
                values: dict[tuple[str, str], list[float]] = defaultdict(list)
                for record in arm_records:
                    value = record.get(field)
                    if record["model"] in {model_a, model_b} and isinstance(value, (bool, int, float)):
                        values[(record["model"], record["case"])].append(float(value))
                common = sorted(
                    {case for model, case in values if model == model_a}
                    & {case for model, case in values if model == model_b}
                )
                diffs = [statistics.fmean(values[(model_b, case)]) - statistics.fmean(values[(model_a, case)]) for case in common]
                if len(diffs) > 1:
                    rng = random.Random(_stable_seed(f"paired:{arm}:{model_a}:{model_b}:{field}"))
                    draws = [statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(BOOTSTRAP_DRAWS)]
                    interval = [_percentile(draws, .025), _percentile(draws, .975)]
                else:
                    interval = [diffs[0], diffs[0]] if diffs else None
                comparison["metrics"][label] = {"common_cases": len(common), "mean_difference": _mean(diffs), "ci95_case_bootstrap": interval}
            output.append(comparison)
    return output


def _group_rollups(records: list[dict], key: str, label: str) -> dict:
    values: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        memberships = record.get(key)
        if not isinstance(memberships, list):
            memberships = [memberships or "unclassified"]
        for membership in memberships:
            values[str(membership)].append(record)
    return {value: _rollup(group, f"{label}:{value}") for value, group in sorted(values.items())}


def build_report(
    experiment: Path, panel: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict]]:
    """Join PV with one judge panel's RQ and CC; with no panel the judged channels are absent."""
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    specs = {spec["name"]: spec for spec in load_specs(BENCHMARKS_JSON)}
    models_seen: dict[str, dict] = {}
    records, review_queue, rq_aggregates = [], [], []
    missing_pv, missing_rq, missing_cc, invalid_results = [], [], [], []
    actual_keys: set[tuple[str, str, str]] = set()

    for case_name, result_path in iter_result_files(experiment):
        _, model_id, run_id = result_identity(result_path)
        identity_base = {"case": case_name, "model": model_id, "run_id": run_id}
        identity = dict(identity_base)
        try:
            saved = load_result(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid_results.append({**identity, "error": str(error)})
            continue
        if not result_is_complete(saved):
            invalid_results.append({**identity, "error": "incomplete_result"})
            continue
        actual_keys.add((case_name, model_id, run_id))
        if isinstance(saved.get("model"), dict):
            models_seen.setdefault(model_id, saved["model"])
        # One record per scored turn: a multi-turn case is several units of
        # measurement, and pooling them would weight a long conversation more
        # heavily than a short one at the same case.
        pv_success, pv_payload = programmatic_verification(result_path)
        pv_turns = {
            (c["id"], t["turn"]): t
            for c in (pv_payload.get("conversations", []) if isinstance(pv_payload, dict) else [])
            for t in c.get("turns", [])
        }
        for conversation in saved["conversations"]:
            for position, result in enumerate(conversation["turns"]):
                identity = {**identity_base, "conversation": conversation["id"],
                            "turn": position + 1}
                turn_key = sample_id(case_name, conversation["id"], position)
                pv_turn = pv_turns.get((conversation["id"], position + 1), {})
                pv_success = pv_turn.get("programmatic_verification", {}).get("success")
                pv_verdict = pv_turn.get("programmatic_verification", {})
                model_metadata = saved.get("model", {})
                input_price = model_metadata.get("input_cost_per_million_usd")
                output_price = model_metadata.get("output_cost_per_million_usd")
                agent_cost = (
                    float(result.get("prompt_tokens", 0) or 0) * input_price / 1_000_000
                    + float(result.get("completion_tokens", 0) or 0) * output_price / 1_000_000
                    if input_price is not None and output_price is not None else None
                )
                snapshot = specs.get(case_name, {"name": case_name})
                arm = str(saved.get("run", {}).get("arm") or "unclassified")
                if pv_success is None:
                    missing_pv.append(identity)
                variable = pv_verdict.get("variable_verification", {})
                if panel is None:
                    (rq_sample, rq_error), (cc_sample, cc_error) = (None, "no_judge_panel"), (None, "no_judge_panel")
                else:
                    rq_sample, rq_error = _sample(channel_path(result_path, "response_quality", panel), RQ_SCHEMA, turn_key)
                    cc_sample, cc_error = _sample(channel_path(result_path, "claim_coverage", panel), CC_SCHEMA, turn_key)
                if rq_error and panel is not None:
                    missing_rq.append({**identity, "reason": rq_error})
                if cc_error and panel is not None:
                    missing_cc.append({**identity, "reason": cc_error})
                rq = rq_sample.get("response_quality") if rq_sample else None
                cc = cc_sample.get("claim_coverage") if cc_sample else None
                if rq:
                    rq_aggregates.append(rq)
                rq_status, cc_status = _panel_status(rq_sample), _panel_status(cc_sample)
                spec_turns = {c.id: c.turns for c in conversations_of(snapshot)}
                turn_claims = spec_turns[conversation["id"]][position].claims
                required = [c["id"] for c in turn_claims if claim_mode(c) == "required"]
                conditional = [c["id"] for c in turn_claims if claim_mode(c) == "conditional"]
                required_rate = conditional_rate = None
                required_missed = False
                if cc:
                    rates = cc.get("coverage_rate", {})
                    required_rate = statistics.fmean(float(rates[item]) for item in required) if required and all(item in rates for item in required) else (1.0 if not required else None)
                    required_missed = bool(cc.get("required_claims_missed_or_tied", []))
                    conditional_rate = len(set(cc.get("conditional_claims_violated_or_tied", []))) / len(conditional) if conditional else 0.0
                rq_overall = float(rq["overall_score"]) if rq else None
                record = {
                    **identity, "arm": arm, "task_family": snapshot.get("task_family", "unclassified"),
                    "source_mode": snapshot.get("source_mode", "unclassified"),
                    "financial_domains": snapshot.get("financial_domains", ["unclassified"]),
                    "result_file": str(result_path.relative_to(experiment)),
                    "pv_passed": pv_success, "pv_numeric": float(pv_success) if pv_success is not None else None,
                    "pv_failure_type": pv_verdict.get("failure_type"),
                    "pv_variable_accuracy": variable.get("accuracy"), "pv_variable_verification": variable or None,
                    "rq_all": rq_overall, "rq_headline": rq_overall if rq_status == "full_primary" else None,
                    "rq_dimensions": rq.get("score_mean_capped") if rq else None,
                    "rq_panel_status": rq_status, "rq_effective_panel": _panel_names(rq_sample.get("effective_panel") if rq_sample else []),
                    "cc_all": required_rate, "cc_headline": required_rate if cc_status == "full_primary" else None,
                    "conditional_violation_rate_all": conditional_rate,
                    "conditional_violation_rate_headline": conditional_rate if cc_status == "full_primary" else None,
                    "conditional_violations": cc.get("conditional_claims_violated_or_tied", []) if cc else None,
                    "cc_panel_status": cc_status, "cc_effective_panel": _panel_names(cc_sample.get("effective_panel") if cc_sample else []),
                    "tables_unreferenced": result.get("tables_unreferenced") or [],
                    "runtime": {key: result.get(key) for key in ("steps", "prompt_tokens", "completion_tokens", "total_tokens", "elapsed")} | {
                        "estimated_cost_usd": agent_cost,
                        "protocol_repair_attempted": bool(isinstance(result.get("protocol_repair"), dict) and result["protocol_repair"].get("attempted")),
                        "protocol_repair_success": bool(isinstance(pv_payload, dict) and pv_payload.get("protocol_repair_verification", {}).get("success")),
                    },
                    "judge_usage": {
                        "response_quality": _judge_usage(rq_sample),
                        "claim_coverage": _judge_usage(cc_sample),
                    },
                }
                for dimension in criteria.dimensions:
                    value = (
                        float(rq.get("score_mean_capped", {}).get(dimension.id))
                        if rq and dimension.id in rq.get("score_mean_capped", {}) else None
                    )
                    record[f"rq_dimension_all:{dimension.id}"] = value
                    record[f"rq_dimension_headline:{dimension.id}"] = (
                        value if rq_status == "full_primary" else None
                    )
                for scenario, weights in COMPOSITE_WEIGHT_SCENARIOS.items():
                    score = composite_score(pv_success, rq_overall, required_rate, required_missed, criteria, weights) \
                        if pv_success is not None and rq_overall is not None and required_rate is not None else None
                    record[f"composite_all:{scenario}"] = score
                    record[f"composite_headline:{scenario}"] = score if rq_status == cc_status == "full_primary" else None
                record["composite"] = record["composite_headline:primary_60_25_15"]
                reasons = _review_reasons(record, rq, cc)
                records.append(record)
                if reasons:
                    review_queue.append({**record, "reasons": reasons})

    infrastructure_errors = []
    for path in sorted((experiment / RUNS_DIR).glob("*/*/*/error.json")):
        try:
            payload = json.loads(path.read_text())
            infrastructure_errors.append({"case": path.parents[2].name, "model": path.parents[1].name,
                                          "run_id": path.parent.name, "error_type": payload.get("error_type"), "error": payload.get("error")})
        except (OSError, json.JSONDecodeError) as error:
            infrastructure_errors.append({"path": str(path), "error": str(error)})
    overall = _rollup(records, "overall")
    agreement = {"schema": AGREEMENT_SCHEMA, "generated_at": utc_now(),
                 **compute_corpus_agreement(rq_aggregates, criteria)}
    primary_composite = overall["headline_full_primary_panel"]["composite"]["primary_60_25_15"]
    report = {
        "schema": RUN_STATS_SCHEMA, "generated_at": utc_now(), "experiment": experiment.name,
        "judge_panel": panel,
        "models": models_seen,
        "statistical_policy": {
            "unit": "model x arm x case after averaging repeat invocations",
            "confidence_interval": "95% deterministic hierarchical bootstrap over case units and repeats",
            "bootstrap_draws": BOOTSTRAP_DRAWS, "headline_panel": "full preregistered primary panel only",
            "nonprimary_panels": "reported only in all-scored sensitivity strata",
        },
        "overall": overall,
        "programmatic_verification": {
            "completed_trajectories": len(records),
            "scored": overall["programmatic_verification"]["n_trajectories"],
            "passed": sum(r.get("pv_passed") is True for r in records),
            "failed": sum(r.get("pv_passed") is False for r in records),
            "pass_rate": overall["programmatic_verification"]["mean"],
            "ci95_hierarchical_bootstrap": overall["programmatic_verification"]["ci95_hierarchical_bootstrap"],
            "variable_accuracy": overall["variable_accuracy"],
        },
        "runtime": overall["runtime"], "judge_usage": overall["judge_usage"],
        "response_quality": {"panel_scope": "full_primary", "scored": overall["headline_full_primary_panel"]["response_quality"]["n_trajectories"],
                             "mean_overall": overall["headline_full_primary_panel"]["response_quality"]["mean"],
                             "ci95_hierarchical_bootstrap": overall["headline_full_primary_panel"]["response_quality"]["ci95_hierarchical_bootstrap"],
                             "by_dimension": overall["headline_full_primary_panel"]["response_quality_by_dimension"]},
        "claim_coverage": {"panel_scope": "full_primary", "scored": overall["headline_full_primary_panel"]["required_claim_coverage"]["n_trajectories"],
                           "mean_required_coverage": overall["headline_full_primary_panel"]["required_claim_coverage"]["mean"],
                           "conditional_violation_rate": overall["headline_full_primary_panel"]["conditional_violation_rate"]},
        "composite": {"panel_scope": "full_primary", "scored": primary_composite["n_trajectories"],
                      "mean": primary_composite["mean"], "weight_sensitivity": overall["headline_full_primary_panel"]["composite"]},
        "by_model": _group_rollups(records, "model", "model"), "by_arm": _group_rollups(records, "arm", "arm"),
        "by_task_family": _group_rollups(records, "task_family", "task"),
        "by_source_mode": _group_rollups(records, "source_mode", "source"),
        "by_financial_domain": _group_rollups(records, "financial_domains", "domain"),
        "paired_model_differences": _paired_differences(records),
        "missing": {"programmatic_verification": missing_pv,
                    "response_quality": missing_rq, "claim_coverage": missing_cc,
                    "invalid_results": invalid_results, "infrastructure_errors": infrastructure_errors},
        "review_queue_size": len(review_queue), "case_scores": records,
    }
    return report, agreement, review_queue


def write_reports(experiment: Path, panel: str | None = None) -> dict[str, Any]:
    """Write one panel's reports to ``aggregates/<panel>/``."""
    report, agreement, review_queue = build_report(experiment, panel)
    aggregates = aggregates_dir(experiment, panel)
    atomic_write_json(aggregates / "run_stats.json", report)
    atomic_write_json(aggregates / "corpus_agreement.json", agreement)
    atomic_write_text(aggregates / "review_queue.jsonl", "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in review_queue
    ))
    return report


def write_all_reports(experiment: Path) -> dict[str | None, dict[str, Any]]:
    """One report per judge panel found in the study, or a PV-only report when there is none."""
    panels: list[str | None] = list(experiment_panels(experiment)) or [None]
    return {panel: write_reports(experiment, panel) for panel in panels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--judge-panel", help="report one judge panel; default: every panel in the study")
    args = parser.parse_args()
    experiment = resolve_experiment(args.experiment)
    if args.judge_panel:
        reports = {args.judge_panel: write_reports(experiment, args.judge_panel)}
    else:
        reports = write_all_reports(experiment)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
