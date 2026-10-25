import importlib
import json

from config import DATASETS_DIR, PROJECT_ROOT
from core.data import runtime_variables


def _expected_outputs(module):
    return {
        variable.name: value
        for variable, value in zip(module.variables, module.ground_truth())
    }


def _claims(spec):
    """Every claim in the case, in turn order: a case may spread them over turns."""
    return [claim for turn in spec["conversations"][0]["turns"] for claim in turn["claims"]]


def test_bis_credit_dsr_joint_leader_anchors_and_country_normalization():
    module = importlib.import_module(
        "evals.comparison_ranking.bis_credit_dsr_joint_leaders"
    )
    truth = module.ground_truth()
    assert truth == (
        32,
        "Hong Kong SAR",
        truth[2],
        "Japan",
        truth[4],
        "Hong Kong SAR",
        truth[6],
        31,
        14,
        18,
    )
    assert round(truth[2], 4) == 332.4073
    assert round(truth[4], 4) == 5.1514
    assert round(truth[6], 1) == 34.1
    expected = _expected_outputs(module)
    assert module.validate(expected).success
    assert module.validate(
        {**expected, "highest_actual_credit_ratio_country": "HK: Hong Kong SAR"}
    ).success
    assert not module.validate(
        {**expected, "actual_ratio_leader_credit_gap_rank": 1}
    ).success


def test_seven_a_negative_employment_screen_anchor_values():
    module = importlib.import_module(
        "evals.comparison_ranking.seven_a_negative_employment_chargeoff_screen"
    )
    truth = module.ground_truth()
    assert truth[:4] == (51, 24, 6, "NY")
    assert round(truth[4], 4) == 3.4247
    assert round(truth[5], 4) == 0.7512
    assert truth[6:9] == ("MD", 655, 19)
    assert round(truth[9], 4) == 2.9008
    assert round(truth[10], 4) == 0.4545
    assert truth[11:13] == (2238248, 2230280)
    assert round(truth[13], 4) == -0.3560
    expected = _expected_outputs(module)
    assert module.validate(expected).success
    assert not module.validate(
        {**expected, "overall_highest_chargeoff_incidence_pct": 3.4275921165}
    ).success


def test_phase8_source_necessity_and_claim_modes_are_explicit():
    bis = json.loads(
        (PROJECT_ROOT / "evals/comparison_ranking/bis_credit_dsr_joint_leaders.json").read_text()
    )
    sba = json.loads(
        (PROJECT_ROOT / "evals/comparison_ranking/seven_a_negative_employment_chargeoff_screen.json").read_text()
    )
    assert bis["data_sources"] == ["bis_credit_gap", "bis_debt_service"]
    assert sba["data_sources"] == [
        "sba_7a",
        "census_geo_crosswalk",
        "bls_qcew",
    ]
    bis_modes = {claim["id"]: claim["mode"] for claim in _claims(bis)}
    assert bis_modes["dsr_measure_semantics"] == "required"
    assert bis_modes["systemic_risk_boundary"] == "conditional"
    sba_modes = {claim["id"]: claim["mode"] for claim in _claims(sba)}
    assert sba_modes["incidence_dollar_loss_share_distinction"] == "required"
    assert sba_modes["time_grain_limitation"] == "required"
    assert sba_modes["employment_causality_boundary"] == "conditional"
    assert sba_modes["qcew_vintage_boundary"] == "conditional"
    assert sba_modes["minimum_population_threshold_boundary"] == "conditional"


def test_sba_runtime_discloses_nonunique_public_row_contract():
    variable = next(
        item
        for item in runtime_variables({"sba_7a"})
        if item.name == "sba_seven_a_loans_df"
    )
    assert "no guaranteed unique public loan identifier" in variable.description
    assert "must not be silently deduplicated" in variable.description
    schema = json.loads(
        (DATASETS_DIR / "sba_7a_504/schema.json").read_text()
    )
    table = schema["tables"]["sba_7a_504.seven_a_loans"]
    assert table["duplicate_policy"]["exact_duplicate_rows_observed"] == 919
    manifest = json.loads(
        (DATASETS_DIR / "sba_7a_504/sources.json").read_text()
    )
    assert "do not provide a guaranteed unique public loan identifier" in manifest[
        "row_identity_notice"
    ]
