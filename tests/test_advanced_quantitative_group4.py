import json
from pathlib import Path

import numpy as np

from evals.scenario_analysis import bis_gaussian_copula_joint_stress as copula
from evals.temporal_path import correlated_release_blue_revision as blue
from scripts.audit_case_design import has_closed_text_representation


ROOT = Path(__file__).resolve().parents[1]


def _truth(module):
    return dict(zip((variable.name for variable in module.variables), module.ground_truth()))


def test_group4_contract_and_turn_validators():
    cases = (
        ("evals/scenario_analysis/bis_gaussian_copula_joint_stress.json", copula),
        ("evals/temporal_path/correlated_release_blue_revision.json", blue),
    )
    for path, module in cases:
        spec = json.loads((ROOT / path).read_text())
        turns = spec["conversations"][0]["turns"]
        assert len(turns) == 4
        assert len(spec["data_sources"]) >= 2
        assert [name for turn in turns for name in turn["stores"]] == [v.name for v in module.variables]
        truth = _truth(module)
        assert all(has_closed_text_representation(v.description) for v,value in zip(module.variables,module.ground_truth()) if isinstance(value,str))
        assert module.validate(truth).success
        for index in range(1, 5):
            names = getattr(module, f"TURN_{index}_NAMES")
            assert getattr(module, f"validate_turn_{index}")({name: truth[name] for name in names}).success


def test_copula_score_and_conditional_identities():
    truth = _truth(copula)
    assert truth["joint_stress_leader"] == "JP"
    contribution_sum = sum(truth[name] for name in (
        "leader_gap_score_contribution", "leader_dsr_score_contribution",
        "leader_depreciation_score_contribution",
    ))
    assert np.isclose(contribution_sum, truth["leader_common_stress_score"])
    assert truth["leader_contribution_sum_residual"] < 1e-12
    assert truth["scenario_common_stress_upper_tail_probability"] < truth["leader_common_stress_upper_tail_probability"]

    candidate = {name: truth[name] for name in copula.TURN_2_NAMES}
    candidate["joint_stress_leader"] = "Japan"
    assert not copula.validate_turn_2(candidate).success


def test_blue_weight_and_comparison_identities():
    truth = _truth(blue)
    assert truth["gdp_selected_covariance_model"] == "full"
    assert truth["gdp_full_covariance_validation_rmse_pp"] < truth["gdp_diagonal_covariance_validation_rmse_pp"]
    assert truth["z1_selected_covariance_model"] == "diagonal"
    assert not truth["z1_selection_agrees_with_gdp"]
    assert np.isclose(sum(truth[name] for name in (
        "gdp_blue_first_weight", "gdp_blue_second_weight", "gdp_blue_third_weight",
    )), 1)
    assert np.isclose(sum(truth[name] for name in (
        "z1_blue_first_weight", "z1_blue_second_weight", "z1_blue_third_weight",
    )), 1)
    assert abs(truth["gdp_third_release_error_pp"]) > abs(truth["gdp_blue_estimate_pct"] - truth["gdp_most_recent_pct"])
    assert abs(truth["z1_third_vintage_error_usd_millions"]) > abs(
        truth["z1_blue_terminal_estimate_usd_millions"] - truth["z1_newest_vintage_usd_millions"]
    )


def test_group4_deterministic_and_wrong_values_rejected():
    for module in (copula, blue):
        first = module.ground_truth()
        module.ground_truth.cache_clear()
        assert first == module.ground_truth()
        truth = _truth(module)
        names = module.TURN_2_NAMES
        candidate = {name: truth[name] for name in names}
        numeric = next(name for name in names if isinstance(candidate[name], (int, float)) and not isinstance(candidate[name], bool))
        candidate[numeric] = float(candidate[numeric]) + 1
        assert not module.validate_turn_2(candidate).success
