import json

from config import PROJECT_ROOT
from core.scoring import Criteria
import scripts.calibrate_judges as calibration


def test_human_calibration_compares_rq_and_claim_judges(monkeypatch, tmp_path):
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    dimensions = [dimension.id for dimension in criteria.dimensions]
    claims = [
        {"id": "required_fact", "mode": "required"},
        {"id": "causal_boundary", "mode": "conditional"},
    ]
    llm_samples = {}
    for index, score in enumerate((2, 4), 1):
        identifier = f"case-{index}/model/run-1"
        llm_samples[identifier] = {
            "case": f"case-{index}", "model": "model", "run_id": "run-1",
            "claims": claims,
            "rq_judges": [
                {"scores": {dimension: value for dimension in dimensions}}
                for value in (score, score + 1)
            ],
            "cc_judges": [
                {"coverage": {"required_fact": True, "causal_boundary": True}},
                {"coverage": {"required_fact": True, "causal_boundary": True}},
            ],
        }
    monkeypatch.setattr(calibration, "load_llm_samples", lambda experiment, panel: llm_samples)
    ratings = tmp_path / "ratings.jsonl"
    rows = []
    for identifier, human_score in zip(llm_samples, (2, 5)):
        for rater in ("finance-a", "finance-b"):
            rows.append({
                "sample_id": identifier, "rater_id": rater,
                "rq_scores": {dimension: human_score for dimension in dimensions},
                "claim_coverage": {"required_fact": True, "causal_boundary": True},
            })
    ratings.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = calibration.build_calibration_report(tmp_path, ratings, "judge-a")
    assert report["coverage"]["human_rated_samples"] == 2
    assert report["coverage"]["samples_with_fewer_than_two_raters"] == []
    dimension = report["response_quality"]["dimensions"][dimensions[0]]
    assert dimension["compared_samples"] == 2
    assert dimension["human_llm_mean_absolute_error"] == 0.5
    assert report["claim_coverage"]["majority_agreement"] == 1.0
    assert report["publication_readiness"]["ready"] is False


def test_human_calibration_rejects_single_or_placeholder_rater(monkeypatch, tmp_path):
    criteria = Criteria.from_json(PROJECT_ROOT / "criteria.json")
    dimensions = [dimension.id for dimension in criteria.dimensions]
    identifier = "case/model/run"
    llm_samples = {identifier: {
        "case": "case", "model": "model", "run_id": "run", "claims": [],
        "rq_judges": [], "cc_judges": [],
    }}
    monkeypatch.setattr(calibration, "load_llm_samples", lambda experiment, panel: llm_samples)
    ratings = tmp_path / "ratings.jsonl"
    ratings.write_text(json.dumps({
        "sample_id": identifier, "rater_id": "REPLACE_WITH_BLINDED_RATER_ID",
        "rq_scores": {dimension: 3 for dimension in dimensions}, "claim_coverage": {},
    }) + "\n")
    try:
        calibration.build_calibration_report(tmp_path, ratings, "judge-a")
    except ValueError as error:
        assert "blinded rater_id" in str(error)
    else:
        raise AssertionError("placeholder rater should be rejected")
