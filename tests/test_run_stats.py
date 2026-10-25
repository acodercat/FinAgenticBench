import json

from config import BENCHMARKS_JSON, PROJECT_ROOT
from core.evaluator import load_specs
from core.models import ModelConfig
from core.results import (
    RESULT_SCHEMA, atomic_write_json, channel_path, panel_label,
    result_path as make_result_path,
)
from core.scoring import Criteria, aggregate_claims, aggregate_quality
from scripts.run_stats import (
    _metric_summary, _panel_status, _runtime_summary, write_reports,
)


def _judge(name):
    return ModelConfig(name=name, api_model=name, api_key="secret", lab=name)


def test_run_stats_joins_three_channels_and_writes_review_queue(tmp_path):
    spec = next(
        item for item in load_specs(BENCHMARKS_JSON)
        if item["name"] == "walmart_fy2025_revenue"
    )
    result_path = make_result_path(tmp_path, spec["name"], "tested", "run-1")
    panel = [{"name": name} for name in ("judge-a", "judge-b", "judge-c")]
    atomic_write_json(result_path, {
        "schema": RESULT_SCHEMA,
        "model": {"name": "tested", "model_id": "tested"},
        "case": spec["name"],
        "conversations": [{"id": "main", "turns": [{
            "turn": 1, "query": spec["conversations"][0]["turns"][0]["query"],
            "stores": spec["conversations"][0]["turns"][0]["stores"],
            "response": "answer", "outputs": {"v": 1},
            "steps": 3, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "elapsed": 2.5, "failure_type": None,
        }]}],
    })
    atomic_write_json(channel_path(result_path, "programmatic_verification"), {
        "schema": "finbench-programmatic-verification-v2",
        "conversations": [{"id": "main", "turns": [
            {"turn": 1, "status": "scored",
             "programmatic_verification": {"success": True}}
        ]}],
    })
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    judge_scores = []
    for name, score in zip(("judge-a", "judge-b", "judge-c"), (3, 4, 5)):
        judge_scores.append({
            "judge_id": name,
            "_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "scores": {dimension.id: score for dimension in criteria.dimensions},
            "justifications": {dimension.id: "evidence" for dimension in criteria.dimensions},
        })
    quality = aggregate_quality(judge_scores, criteria)
    coverage = aggregate_claims([{
        "judge_id": name,
        "_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        "coverage": {claim["id"]: True for claim in spec["conversations"][0]["turns"][0]["claims"]},
        "justifications": {claim["id"]: "explicit" for claim in spec["conversations"][0]["turns"][0]["claims"]},
    } for name in ("judge-a", "judge-b", "judge-c")], spec["conversations"][0]["turns"][0]["claims"])
    judges = panel_label(["judge-a", "judge-b", "judge-c"])
    atomic_write_json(channel_path(result_path, "response_quality", judges), {
        "schema": "finbench-response-quality-v2",
        "conversations": [{"id": "main", "turns": [{
            "turn": 1, "sample_id": f"{spec['name']}/main/0", "status": "scored",
            "response_quality": quality, "requested_panel": panel, "effective_panel": panel,
            "judge_failures": [],
        }]}],
    })
    atomic_write_json(channel_path(result_path, "claim_coverage", judges), {
        "schema": "finbench-claim-coverage-v2",
        "conversations": [{"id": "main", "turns": [{
            "turn": 1, "sample_id": f"{spec['name']}/main/0", "status": "scored",
            "claim_coverage": coverage, "requested_panel": panel, "effective_panel": panel,
            "judge_failures": [],
        }]}],
    })

    report = write_reports(tmp_path, judges)
    assert report["judge_panel"] == judges
    assert report["programmatic_verification"]["pass_rate"] == 1.0
    assert report["response_quality"]["mean_overall"] == 4.0
    assert report["claim_coverage"]["mean_required_coverage"] == 1.0
    assert report["composite"]["mean"] == 93.75
    assert report["runtime"]["total_tokens"] == 15
    assert report["judge_usage"]["total_tokens"] == 66
    assert report["runtime"]["mean_steps"] == 3
    assert report["programmatic_verification"]["completed_trajectories"] == 1
    assert report["by_task_family"][spec["task_family"]]["pv_pass_rate"] == 1.0
    assert report["by_task_family"][spec["task_family"]]["rq_mean_full_primary"] == 4.0
    assert report["by_task_family"][spec["task_family"]][
        "required_claim_coverage_full_primary"
    ] == 1.0
    assert report["by_source_mode"][spec["source_mode"]]["pv_pass_rate"] == 1.0
    for domain in spec["financial_domains"]:
        assert report["by_financial_domain"][domain]["pv_pass_rate"] == 1.0
    assert report["review_queue_size"] == 1
    reports = tmp_path / "aggregates" / judges
    assert json.loads((reports / "review_queue.jsonl").read_text())["case"] == spec["name"]
    agreement = json.loads((reports / "corpus_agreement.json").read_text())
    assert agreement["dimensions"]["communication"]["n_ratings"] == 3


def test_run_stats_repeat_policy_uses_case_as_statistical_unit():
    records = [
        {"model": "m", "arm": "strict", "case": "a", "metric": 0},
        {"model": "m", "arm": "strict", "case": "a", "metric": 1},
        {"model": "m", "arm": "strict", "case": "b", "metric": 1},
    ]
    summary = _metric_summary(records, "metric", label="repeat-policy-test")
    assert summary["n_trajectories"] == 3
    assert summary["n_case_units"] == 2
    assert summary["mean"] == 0.75
    assert summary["ci95_hierarchical_bootstrap"] is not None


def test_panel_status_separates_full_and_partial_panels():
    requested = [{"name": name} for name in ("a", "b", "c")]
    assert _panel_status({
        "requested_panel": requested,
        "effective_panel": requested,
    }) == "full_primary"
    assert _panel_status({
        "requested_panel": requested,
        "effective_panel": [{"name": "a"}, {"name": "b"}],
    }) == "partial_primary"


def test_failure_taxonomy_uses_the_authoritative_channel():
    """A stale runtime verdict must not enter the taxonomy beside the PV verdict."""
    runtime = _runtime_summary([{
        "case": "a", "model": "provider-model", "run_id": "repeat-1", "arm": "strict",
        "pv_failure_type": "wrong_value",
        "runtime": {"failure_type": "stale_trajectory_value"},
    }])
    assert runtime["failure_types"] == {"wrong_value": 1}
