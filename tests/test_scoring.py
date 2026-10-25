import pytest

from config import PROJECT_ROOT
from core.scoring import (
    pass_hat_k,
    COMPOSITE_WEIGHT_SCENARIOS, Criteria, aggregate_claims, aggregate_quality,
    apply_dependency_cap, composite_score, compute_corpus_agreement,
    krippendorff_alpha,
)


CRITERIA = Criteria.from_json(PROJECT_ROOT / "criteria.json")


def test_dependency_cap_prevents_polished_downstream_scores_hiding_bad_method():
    scores = {dimension.id: 5 for dimension in CRITERIA.dimensions}
    scores["analytical_method"] = 2
    capped = apply_dependency_cap(scores, CRITERIA)
    assert capped["analytical_method"] == 2
    assert capped["evidence_grounding"] == 3
    assert capped["interpretation_and_calibration"] == 3
    assert capped["communication"] == 3


def test_three_channel_composite_has_hard_gates():
    assert composite_score(False, 5, 1, False, CRITERIA) == 40
    assert composite_score(True, 5, 1, False, CRITERIA) == 100
    assert composite_score(True, 5, 0, True, CRITERIA) == 75
    assert composite_score(
        True, 5, 1, False, CRITERIA,
        COMPOSITE_WEIGHT_SCENARIOS["pv_heavy_70_20_10"],
    ) == 100
    with pytest.raises(ValueError, match="sum to 1"):
        composite_score(True, 5, 1, False, CRITERIA, {
            "programmatic_verification": 1,
            "response_quality": 1,
            "claim_coverage": 1,
        })


def test_quality_and_claim_aggregation_preserve_diagnostics():
    scores = {dimension.id: 4 for dimension in CRITERIA.dimensions}
    quality = aggregate_quality([{
        "scores": scores,
        "justifications": {dimension.id: "specific evidence" for dimension in CRITERIA.dimensions},
    }], CRITERIA)
    assert quality["overall_score"] == pytest.approx(4)
    claims = [{"id": "context", "description": "x", "mode": "required"}]
    coverage = aggregate_claims(
        [{"coverage": {"context": False}, "justifications": {"context": "absent"}}], claims
    )
    assert coverage["flagged_for_review"]


def test_optional_absence_does_not_flag_and_conditional_violation_does():
    claims = [
        {"id": "detail", "description": "x", "mode": "optional"},
        {"id": "boundary", "description": "y", "mode": "conditional"},
    ]
    clean = aggregate_claims([{
        "coverage": {"detail": False, "boundary": True},
        "justifications": {"detail": "absent", "boundary": "not violated"},
    }], claims)
    assert not clean["flagged_for_review"]
    violated = aggregate_claims([{
        "coverage": {"detail": False, "boundary": False},
        "justifications": {"detail": "absent", "boundary": "violated"},
    }], claims)
    assert violated["conditional_claims_violated_or_tied"] == ["boundary"]
    assert violated["flagged_for_review"]


def test_two_anchor_judge_range_is_flagged_even_below_std_threshold():
    low = {dimension.id: 3 for dimension in CRITERIA.dimensions}
    high = {dimension.id: 5 for dimension in CRITERIA.dimensions}
    judges = [
        {"scores": scores, "justifications": {
            dimension.id: "specific evidence" for dimension in CRITERIA.dimensions
        }}
        for scores in (low, high, high)
    ]
    quality = aggregate_quality(judges, CRITERIA)
    assert quality["score_std"]["scope_and_conventions"] < 1
    assert quality["score_range"]["scope_and_conventions"] == 2
    assert quality["flagged_for_review"]


def test_a_cap_at_or_below_its_own_trigger_is_rejected():
    """A cap that re-triggers itself makes the rule depend on evaluation order."""
    for downstream_cap in (2, 1):
        broken = Criteria(
            CRITERIA.dimensions, CRITERIA.channel_weights,
            {**CRITERIA.gates, "downstream_cap": downstream_cap},
        )
        with pytest.raises(ValueError, match="must exceed dependency_trigger"):
            broken.validate()


def test_flag_reasons_name_the_extremes_the_bare_flag_would_hide():
    def judged(level):
        return {
            "scores": {dimension.id: level for dimension in CRITERIA.dimensions},
            "justifications": {
                dimension.id: "specific evidence" for dimension in CRITERIA.dimensions
            },
        }

    floor = aggregate_quality([judged(1), judged(1)], CRITERIA)
    assert any(
        reason.startswith("extreme_low:") for reason in floor["flag_reasons"]
    )
    ceiling = aggregate_quality([judged(5), judged(5)], CRITERIA)
    assert any(
        reason.startswith("extreme_high:") for reason in ceiling["flag_reasons"]
    )
    # A mid-range consensus is neither, and carries no reasons at all.
    middle = aggregate_quality([judged(3), judged(3)], CRITERIA)
    assert middle["flag_reasons"] == [] and not middle["flagged_for_review"]


def test_a_split_panel_is_reported_as_a_tie_not_as_a_miss():
    claims = [{"id": "context", "description": "x", "mode": "required"}]
    tied = aggregate_claims([
        {"coverage": {"context": True}, "justifications": {"context": "stated"}},
        {"coverage": {"context": False}, "justifications": {"context": "absent"}},
    ], claims)
    assert tied["required_claims_tied"] == ["context"]
    assert tied["required_claims_missed"] == []
    # The composite gate still fires on the combined list.
    assert tied["required_claims_missed_or_tied"] == ["context"]
    assert tied["flag_reasons"] == ["required_claim_tie:context"]

    missed = aggregate_claims([
        {"coverage": {"context": False}, "justifications": {"context": "absent"}},
        {"coverage": {"context": False}, "justifications": {"context": "absent"}},
    ], claims)
    assert missed["required_claims_missed"] == ["context"]
    assert missed["required_claims_tied"] == []
    assert missed["flag_reasons"] == ["required_claim_missed:context"]


def test_krippendorff_agreement_is_corpus_level_and_supports_two_metrics():
    assert krippendorff_alpha([[1, 1], [2, 2]], "interval") == 1.0
    assert krippendorff_alpha([[1, 2], [1, 2]], "nominal") == pytest.approx(-0.5)
    assert krippendorff_alpha([[1]], "interval") is None

    scores = {dimension.id: 4 for dimension in CRITERIA.dimensions}
    sample = aggregate_quality([{
        "scores": scores,
        "justifications": {dimension.id: "evidence" for dimension in CRITERIA.dimensions},
    }, {
        "scores": scores,
        "justifications": {dimension.id: "evidence" for dimension in CRITERIA.dimensions},
    }], CRITERIA)
    agreement = compute_corpus_agreement([sample], CRITERIA)
    assert agreement["dimensions"]["analytical_method"] == {
        "n_units": 1, "n_ratings": 2,
        "interval_alpha": 1.0, "nominal_alpha": 1.0,
    }


def test_pass_hat_k_separates_reliability_from_mean_accuracy():
    # Two of three trials pass, so the mean is 2/3, but the case never passes
    # all three times and pass^3 is zero.
    assert pass_hat_k(3, 2, 1) == pytest.approx(2 / 3)
    assert pass_hat_k(3, 2, 2) == pytest.approx(1 / 3)
    assert pass_hat_k(3, 2, 3) == 0.0
    assert pass_hat_k(3, 3, 3) == 1.0

    with pytest.raises(ValueError):
        pass_hat_k(2, 2, 3)
    with pytest.raises(ValueError):
        pass_hat_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 0)


def test_reliability_excludes_cases_with_too_few_trials():
    from scripts.run_stats import _reliability

    records = [
        {"model": "m", "arm": "a", "case": "always", "pv_passed": True},
        {"model": "m", "arm": "a", "case": "always", "pv_passed": True},
        {"model": "m", "arm": "a", "case": "flaky", "pv_passed": True},
        {"model": "m", "arm": "a", "case": "flaky", "pv_passed": False},
        {"model": "m", "arm": "a", "case": "once", "pv_passed": True},
    ]
    summary = _reliability(records)
    assert summary["cases"] == 3
    assert summary["trials_per_case"] == {1: 1, 2: 2}
    assert summary["pass_hat_k"]["1"] == {"value": pytest.approx(5 / 6), "cases": 3}
    # `once` has a single trial, so it cannot answer the pass^2 question.
    assert summary["pass_hat_k"]["2"] == {"value": pytest.approx(0.5), "cases": 2}

